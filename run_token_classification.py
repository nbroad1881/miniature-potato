import os
import datetime
import argparse
from functools import partial

import wandb
import torch
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer_utils import set_seed
from transformers.integrations import WandbCallback


from callbacks import NewWandbCB, SaveCallback
from utils import (
    get_configs,
    set_wandb_env_vars,
    reinit_model_weights,
    create_optimizer,
    create_scheduler,
    push_to_hub,
)
from data import (
    TokenClassificationDataModule,
    AI4CodeDataCollator,
)
from modeling import (
    get_pretrained,
)
from metrics import ai4code_compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on AI4Code dataset")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Config file",
    )
    parser.add_argument(
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
    dm = TokenClassificationDataModule(cfg)

    dm.prepare_datasets()

    for fold in range(3):

        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        cfg["output"] = output
        cfg["load_from_disk"] = load_from_disk
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        metric_to_track = "eval_kt"
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

        compute_metrics = partial(
            ai4code_compute_metrics,
            eval_dataset=eval_dataset,
        )

        model_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )
        model_config.update(
            {
                "num_labels": 1,
                "output_dropout_prob": cfg["dropout"],
                # "attention_probs_dropout_prob": 0.0,
                # "hidden_dropout_prob": 0.0,
                "multisample_dropout": cfg["multisample_dropout"],
                # "layer_norm_eps": cfg["layer_norm_eps"],
                "run_start": str(datetime.datetime.utcnow()),
                "output_layer_norm": cfg["output_layer_norm"],
                "cls_token_map": dm.cls_id_map,
            }
        )

        model = get_pretrained(model_config, cfg["model_name_or_path"])
        model.backbone.resize_token_embeddings(len(dm.tokenizer))

        reinit_model_weights(model, cfg["reinit_layers"], model_config)

        optimizer = create_optimizer(model, args)

        steps_per_epoch = (
            len(train_dataset)
            // args.per_device_train_batch_size
            // cfg["n_gpu"]
            // args.gradient_accumulation_steps
        )
        num_training_steps = steps_per_epoch * args.num_train_epochs

        scheduler = create_scheduler(num_training_steps, optimizer, args)

        collator = AI4CodeDataCollator(
            tokenizer=dm.tokenizer, pad_to_multiple_of=cfg["pad_multiple"], padding=True
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=dm.tokenizer,
            callbacks=callbacks,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
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
