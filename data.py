import os
import json
import pickle
from pathlib import Path
from itertools import chain
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)
from datasets import Dataset, disable_progress_bar, load_dataset, load_from_disk


@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError("Please provide a config file")

        self.data_dir = Path(self.cfg["data_dir"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

        self.cls_tkn_map = {"code": "[CLS_CODE]", "markdown": "[CLS_MARKDOWN]"}

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.cls_tkn_map.values())}
        )
        self.cls_id_map = {
            "code": self.tokenizer.encode(self.cls_tkn_map["code"])[1],
            "markdown": self.tokenizer.encode(self.cls_tkn_map["markdown"])[1],
        }

        if self.cfg["load_from_disk"]:
            if self.cfg["load_from_disk"].endswith(".dataset"):
                load_path = self.cfg["load_from_disk"][: -len(".dataset")]
            else:
                load_path = self.cfg["load_from_disk"]
            self.ds = load_from_disk(f"{load_path}.dataset")

            with open(f"{load_path}.pkl", "rb") as fp:
                self.fold_idxs = pickle.load(fp)
            print("Loading dataset from disk", self.cfg["load_from_disk"])
            if self.cfg["DEBUG"]:
                self.ds = self.ds.select(range(1000))
        else:
            self.orders_ds = load_dataset(
                "csv", data_files=str(self.data_dir / "train_orders.csv"), split="train"
            )

            if self.cfg["DEBUG"]:
                self.orders_ds = self.orders_ds.select(range(10_000))

            self.orders_ds = self.orders_ds.map(
                lambda x: {"length": [len(x.split()) for x in x["cell_order"]]},
                batched=True,
                num_proc=self.cfg["num_proc"],
                desc="Calculating length of cell orders",
            )

            self.fold_idxs = get_folds(
                self.orders_ds.to_pandas(),
                pd.read_csv(self.data_dir / "train_ancestors.csv"),
                k_folds=self.cfg["k_folds"],
                stratify_on=self.cfg["stratify_on"],
                groups=self.cfg["fold_groups"],
            )

    def prepare_datasets(self):

        if self.cfg["load_from_disk"] is None:

            self.ds = self.orders_ds.map(
                self.read_text_files, batched=False, num_proc=self.cfg["num_proc"]
            )

            print(sum(self.ds["error"]), "errors")

            # disable_progress_bar()

            self.ds = self.ds.map(
                self.tokenize,
                batched=False,
                num_proc=self.cfg["num_proc"],
                desc="Tokenizing",
            )

            self.ds = self.ds.map(
                self.add_cls_tokens,
                batched=False,
                num_proc=self.cfg["num_proc"],
                desc="Adding CLS tokens",
                fn_kwargs={"max_length":self.cfg["max_length"]}
            )

            self.ds.save_to_disk(f"{self.cfg['output']}.dataset")
            with open(f"{self.cfg['output']}.pkl", "wb") as fp:
                pickle.dump(self.fold_idxs, fp)

            print("Saving dataset to disk:", self.cfg["output"])

    def get_train_dataset(self, fold):
        idxs = list(chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold]))
        return self.ds.select(idxs)

    def get_eval_dataset(self, fold):
        idxs = self.fold_idxs[fold]
        # print("Unique eval fold values:", self.raw_ds.select(idxs).unique("fold"))
        return self.ds.select(idxs)

    def read_text_files(self, example):

        id_ = example["id"]

        example["error"] = False

        with open(self.data_dir / "train" / f"{id_}.json", "r") as fp:
            data = json.load(fp)
            try:
                example["source"] = data["source"].values()
                example["cell_type"] = data["cell_type"].values()

                example["cell_ids"] = list(data["source"].keys())
                example["correct_order"] = example["cell_order"].split()
                cell_id2idx = {
                    cell_id: example["correct_order"].index(cell_id)
                    for cell_id in example["cell_ids"]
                }

                num_cells = len(example["cell_type"])
                example["labels"] = [
                    cell_id2idx[cell_id] / num_cells
                    for cell_id in data["source"].keys()
                ]

            except:
                example["error"] = True

        return example

    def tokenize(self, example):

        tokenized = self.tokenizer(
            example["source"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )

        total_tokens = len(list(chain(*tokenized["input_ids"])))

        # get rid of first cls and last eos
        input_ids = [x[1:-1] for x in tokenized["input_ids"]]

        new_ids = []
        max_length = self.cfg["max_length"]
        if total_tokens > (max_length - len(example["source"])):
            markdown_idxs = set(
                [
                    i
                    for i, cell_type in enumerate(example["cell_type"])
                    if cell_type == "markdown"
                ]
            )
            markdown_length = len(
                list(
                    chain(
                        [
                            x
                            for i, x in enumerate(tokenized["input_ids"])
                            if i in markdown_idxs
                        ]
                    )
                )
            )
            chunk_size = (max_length - len(example["source"])) // len(example["source"])
            new_ids = [x[:chunk_size] for x in input_ids]
            return {"input_id_list": new_ids}

        return {"input_id_list": input_ids}

    def add_cls_tokens(self, example, max_length=1024):
        new_ids = []

        for cell_type, ids in zip(example["cell_type"], example["input_id_list"]):
            new_ids.extend([self.cls_id_map[cell_type]] + ids)

        attention_mask = (
            [1] * len(new_ids) if len(new_ids) < max_length else [1] * max_length
        )

        return {
            "input_ids": new_ids[:max_length],
            "attention_mask": attention_mask,
        }


def get_folds(
    orders_df, ancestors_df, k_folds=5, stratify_on="length", groups="ancestor_id"
):

    merged = orders_df.merge(ancestors_df, on="id", how="left")

    if stratify_on and groups:
        sgkf = StratifiedGroupKFold(n_splits=k_folds)
        return [
            val_idx
            for _, val_idx in sgkf.split(
                merged, y=merged[stratify_on].astype(str), groups=merged[groups]
            )
        ]

    gkf = GroupKFold(n_splits=k_folds)
    return [val_idx for _, val_idx in gkf.split(merged, groups=merged[groups])]


@dataclass
class OnlyMaskingCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>
    
    See original source code here: https://github.com/huggingface/transformers/blob/8b332a6a160c6df82e4267aaf118d87377d78a67/src/transformers/data/data_collator.py#L607
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # IGNORE RANDOM/NO MASK
        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels


@dataclass
class AI4CodeDataCollator(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    DataCollatorForTokenClassification does not have floats for the labels. 
    This must be used with the MaskingProbCallback that sets the environment
    variable at the beginning and end of the training step. This callback ensures
    that there is no masking done during evaluation.

    see data collator source code here: https://github.com/huggingface/transformers/blob/8b332a6a160c6df82e4267aaf118d87377d78a67/src/transformers/data/data_collator.py#L264 
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        labels = torch.tensor(batch[label_name], dtype=torch.float32)
        batch = {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batch.items()
            if k != label_name
        }
        batch[label_name] = labels

        masking_prob = os.getenv("MASKING_PROB")
        if masking_prob is not None and masking_prob != "0":
            batch = self.mask_tokens(batch, float(masking_prob))

        return batch
