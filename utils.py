import os
import re
import yaml
import json
from pathlib import Path

from itertools import chain
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple

import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error

from transformers import (
    get_scheduler,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
)
from transformers.utils import logging
import bitsandbytes as bnb
from modelcards import CardData, ModelCard

logger = logging.get_logger(__name__)


def fix_e(cfg):
    def fix(value):
        pattern = r"\d+e\-\d+"
        if re.search(pattern, value):
            return eval(value)
        return value

    for k, v in cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    cfg[k][kk] = fix(vv)
        elif isinstance(v, str):
            cfg[k] = fix(v)

    return cfg


def remove_defaults(cfg):
    to_remove = []
    args = cfg["training_arguments"]
    for key, value in args.items():
        if value == "<default>":
            to_remove.append(key)

    for key in to_remove:
        del args[key]


def get_configs(filename, filepath="./configs"):

    file = Path(filepath) / filename
    with open(file) as fp:
        cfg = yaml.safe_load(fp)

    remove_defaults(cfg)
    cfg = fix_e(cfg)

    # cfg["training_arguments"]["dataloader_num_workers"] = cfg["num_proc"]

    training_args = cfg.pop("training_arguments")
    return cfg, training_args


def set_wandb_env_vars(cfg):
    os.environ["WANDB_ENTITY"] = cfg.get("entity", "")
    os.environ["WANDB_PROJECT"] = cfg.get("project", "")
    os.environ["WANDB_RUN_GROUP"] = cfg.get("group", "")
    os.environ["WANDB_JOB_TYPE"] = cfg.get("job_type", "")
    os.environ["WANDB_NOTES"] = cfg.get("notes", "")
    os.environ["WANDB_TAGS"] = ",".join(cfg.get("tags", ""))


def reinit_model_weights(model, n_layers, config):

    backbone = getattr(model, "backbone", model)
    if "bart" in config.model_type:
        std = config.init_std
    else:
        std = config.initializer_range

    if n_layers > 0:
        if config.model_type == "bart":
            encoder_layers = backbone.encoder.layers
            decoder_layers = backbone.decoder.layers

            reinit_layers(encoder_layers, n_layers, std)
            reinit_layers(decoder_layers, n_layers, std)
        else:
            encoder_layers = backbone.encoder.layer
            reinit_layers(encoder_layers, n_layers, std)


def reinit_layers(layers, n_layers, std):
    for layer in layers[-n_layers:]:
        reinit_modules(layer.modules(), std)


def reinit_modules(modules, std, reinit_embeddings=False):
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif reinit_embeddings and isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def layerwise_learning_rate(model, lr=3e-5, wd=0.01, alpha=0.8):
    model_type = model.backbone_name

    layers = (
        [getattr(model, model_type).embeddings]
        + [getattr(model, model_type).encoder.layer]
        + [model.output]
    )
    layers.reverse()

    optimizer_grouped_parameters = []

    for i, layer in enumerate(layers):
        # This keeps top layer = lr
        if i > 0:
            lr *= alpha
        optimizer_grouped_parameters += uniform_learning_rate(layer, wd)

    return optimizer_grouped_parameters


def create_optimizer(model, train_args, use_8bit=True):

    if use_8bit:
        adam = bnb.optim.Adam8bit
    else:
        adam = bnb.optim.Adam32bit

    opt = adam(
        uniform_learning_rate(model, train_args.learning_rate, train_args.weight_decay),
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon,
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                module, "weight", {"optim_bits": 32}
            )

    return opt


def create_scheduler(num_training_steps, optimizer, train_args, **kwargs):

    if train_args.warmup_ratio > 0:
        warmup_steps = num_training_steps * train_args.warmup_ratio
    else:
        warmup_steps = train_args.warmup_steps

    scheduler = get_scheduler(
        train_args.lr_scheduler_type,
        optimizer,
        warmup_steps,
        num_training_steps,
    )

    return scheduler


def uniform_learning_rate(model, lr, wd=0.01):

    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
            "lr": lr * 0.5,
        },
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr * 0.5,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "weight_decay": 0.0,
            "lr": lr * 1.5,
        },
    ]


def freeze_layers(model, n_layers, freeze_embeds=True):
    if freeze_embeds:
        model.embeddings.requires_grad_(False)

    model.encoder.layer[:n_layers].requires_grad_(False)


def log_training_dynamics(
    output_dir: os.path,
    epoch: int,
    train_ids: List[int],
    train_probas: List[List[float]],
    train_golds: List[int],
):
    """
    For dataset cartography
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """

    td_df = pd.DataFrame(
        {"guid": train_ids, f"logits_epoch_{epoch}": train_probas, "gold": train_golds}
    )

    logging_dir = os.path.join(output_dir, f"training_dynamics")
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Training Dynamics logged to {epoch_file_name}")


def push_to_hub(
    trainer,
    commit_message: Optional[str] = "End of training",
    blocking: bool = True,
    config: str = None,
    metrics: dict = None,   
    wandb_run_id: str = None,
    **kwargs,
) -> str:
    """
    Upload *self.model* and *self.tokenizer* to the 🤗 model hub on the repo *self.args.hub_model_id*.
    Parameters:
        commit_message (`str`, *optional*, defaults to `"End of training"`):
            Message to commit while pushing.
        blocking (`bool`, *optional*, defaults to `True`):
            Whether the function should return only when the `git push` has finished.
        kwargs:
            Additional keyword arguments passed along to [`~Trainer.create_model_card`].
    Returns:
        The url of the commit of your model in the given repository if `blocking=False`, a tuple with the url of
        the commit and an object to track the progress of the commit if `blocking=True`
    """
    # If a user calls manually `push_to_hub` with `self.args.push_to_hub = False`, we try to create the repo but
    # it might fail.
    if not hasattr(trainer, "repo"):
        trainer.init_git_repo()

    # Only push from one node.
    if not trainer.is_world_process_zero():
        return

    if trainer.args.hub_model_id is None:
        model_name = Path(trainer.args.output_dir).name
    else:
        model_name = trainer.args.hub_model_id.split("/")[-1]

    # Cancel any async push in progress if blocking=True. The commits will all be pushed together.
    if (
        blocking
        and trainer.push_in_progress is not None
        and not trainer.push_in_progress.is_done
    ):
        trainer.push_in_progress._process.kill()
        trainer.push_in_progress = None

    git_head_commit_url = trainer.repo.push_to_hub(
        commit_message=commit_message, blocking=blocking, auto_lfs_prune=True
    )
    # push separately the model card to be independant from the rest of the model
    if trainer.args.should_save:
        
        model_card = create_model_card(config, metrics, wandb_run_id)
        model_card.save(Path(trainer.args.output_dir)/"README.md")

        try:
            trainer.repo.push_to_hub(
                commit_message="update model card README.md",
                blocking=blocking,
                auto_lfs_prune=True,
            )
        except EnvironmentError as exc:
            print(
                f"Error pushing update to the model card. Please read logs and retry.\n${exc}"
            )

    return git_head_commit_url


def create_model_card(config, metrics, wandb_run_id):
    """
    config (Dict)
    metrics (Dict)
    wandb_run_id (str)
    """

    template_path = Path(__file__).resolve().parent / "modelcard_template.md"

    return ModelCard.from_template(
        card_data=CardData(  # Card metadata object that will be converted to YAML block
            language='en',
            license='mit',
            tags=['ai4code']+config["tags"],
            datasets=config["dataset_name"],
        ),
        template_path=template_path, 
        model_id=f"{config['output']}-f{config['fold']}",  
        dataset_name=config["dataset_name"], 
        metrics=json.dumps(metrics, indent=4),
        config=json.dumps(config, indent=4),
        wandb_run_id=wandb_run_id,
    )

    