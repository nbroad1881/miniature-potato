import os
import json
import pickle
from pathlib import Path
from itertools import chain
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
)
from transformers import AutoTokenizer
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
                load_path = self.cfg["load_from_disk"][:-len(".dataset")]
            else:
                load_path = self.cfg["load_from_disk"] 
            self.ds = load_from_disk(f'{load_path}.dataset')

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
                self.orders_ds = self.orders_ds.select(range(1000))

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
            )

            self.ds.save_to_disk(f"{self.cfg['output']}.dataset")
            with open(f"{self.cfg['output']}.pkl", "wb") as fp:
                pickle.dump(self.fold_idxs, fp)
        
            print("Saving dataset to disk:", self.cfg['output'])

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
            add_special_tokens=False,
        )

        total_tokens = len(list(chain(*tokenized["input_ids"])))

        new_ids = []
        max_length = self.cfg["max_length"]
        if total_tokens > (max_length - len(example["source"])):
            chunk_size = (max_length - len(example["source"])) // len(example["source"])
            new_ids = [x[:chunk_size] for x in tokenized["input_ids"]]
            return {"input_id_list": new_ids}

        return {"input_id_list": tokenized["input_ids"]}

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
