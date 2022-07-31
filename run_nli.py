import os
import datetime
import argparse

import wandb
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback


from data import NLIDataModule, SwapDataModule
from callbacks import NewWandbCB, SaveCallback
from utils import (
    get_configs,
    set_wandb_env_vars,
    reinit_model_weights,
    push_to_hub,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on AI4Code dataset")
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file",
    )
    parser.add_argument(
        "-l",
        "--load_from_disk",
        type=str,
        required=False,
        default=None,
        help="path to saved dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config_file
    load_from_disk = args.load_from_disk

    output = config_file.split(".")[0]
    cfg, args = get_configs(config_file)
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    cfg["output"] = output
    cfg["load_from_disk"] = load_from_disk

    if cfg["nli"]:
        dm = NLIDataModule(cfg)
    elif cfg["swap"]:
        dm = SwapDataModule(cfg)

    dm.prepare_datasets()


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        all_preds = p.predictions.argmax(-1)

        labels = p.label_ids

        def get_acc(a, b):
            return (a==b).astype(np.float32).mean().item()

        label_scores = {}
        for label_name, val in dm.label2idx.items():
            label_scores[f"{label_name}_accuracy"] = get_acc(all_preds[labels==val], labels[labels==val])


        return {
            "overall_accuracy": get_acc(all_preds, labels),
            **label_scores
            }

    for fold in range(cfg["k_folds"]):

        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        cfg["output"] = output
        cfg["load_from_disk"] = load_from_disk
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        metric_to_track = "eval_overall_accuracy"
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"],
            metric_name=metric_to_track,
            weights_only=True,
        )

        callbacks = [wb_callback, save_callback]

        train_dataset = dm.get_train_dataset(fold)
        eval_dataset = dm.get_eval_dataset(fold)
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Eval dataset length: {len(eval_dataset)}")

        print(
            "Decode inputs from train_dataset",
            dm.tokenizer.convert_ids_to_tokens(train_dataset[0]["input_ids"]),
        )
        print(
            "Decode inputs from train_dataset",
            dm.tokenizer.decode(train_dataset[0]["input_ids"]),
        )


        model_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )
        model_config.update(
            {
                "num_labels": len(dm.label2idx.values()),
                "label2idx": dm.label2idx,
                "idx2label": {v:k for k, v in dm.label2idx.items()},
                # "attention_probs_dropout_prob": 0.0,
                # "hidden_dropout_prob": 0.0,
                # "multisample_dropout": cfg["multisample_dropout"],
                # "layer_norm_eps": cfg["layer_norm_eps"],
                "run_start": str(datetime.datetime.utcnow()),
                # "output_layer_norm": cfg["output_layer_norm"],
            }
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name_or_path"],
            config=model_config
        )
        model.resize_token_embeddings(len(dm.tokenizer))

        reinit_model_weights(model, cfg["reinit_layers"], model_config)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=dm.tokenizer,
            callbacks=callbacks,
        )

        trainer.remove_callback(WandbCallback)

        trainer.train()

        best_metric_score = trainer.model.config.to_dict().get(
            f"best_{metric_to_track}"
        )
        trainer.log({f"best_{metric_to_track}": best_metric_score})
        model.config.update({"wandb_id": wandb.run.id, "wandb_name": wandb.run.name})
        model.config.save_pretrained(args.output_dir)

        if args.push_to_hub:
            push_to_hub(
                trainer,
                config=cfg,
                metrics={f"best_{metric_to_track}": best_metric_score},
                wandb_run_id=wandb.run.id
            )

        wandb.finish()

        torch.cuda.empty_cache()