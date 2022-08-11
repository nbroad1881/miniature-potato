import os
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
import numpy as np
from transformers import (
    PreTrainedModel,
    AutoModel,
)
from transformers.utils import ModelOutput


class AI4CodeModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = AutoModel.from_config(config)

        self.dropout = nn.Dropout(config.output_dropout_prob)
        if config.multisample_dropout:
            self.multisample_dropout = MultiSampleDropout(config.multisample_dropout)

        self.ln = nn.Identity()
        if config.output_layer_norm:
            self.ln = nn.LayerNorm(config.hidden_size)
            self._init_weights(self.ln)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.classifier)

        self.cls_token_ids = torch.tensor(list(config.cls_token_map.values()))


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs,
    ):

        token_type_ids = (
            {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **token_type_ids,
            **kwargs,
        )[0]

        mask = torch.isin(input_ids, self.cls_token_ids.to(input_ids.device))
        loss = None
        if labels is not None:

            labels = labels[labels > -1]
            
            loss_fct = nn.L1Loss()

            if self.config.multisample_dropout:
                loss, logits = self.multisample_dropout(outputs, self.classifier, labels, loss_fct, self.ln, mask)
            else:

                logits = self.classifier(self.ln(self.dropout(outputs)))

                loss = loss_fct(logits[mask].view(-1), labels.view(-1))

        else:
            logits = self.classifier(self.ln(outputs))
            


        return ClassifierOutput(
            loss=loss, logits=logits,
        )

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def get_pretrained(config, model_path):
    model = AI4CodeModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    else:
        model.backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

    return model


@dataclass
class ClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_probs) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]

    def forward(self, hidden_states, linear, labels, loss_fn, layer_nm, mask):
        # if not using output layer_nm, pass nn.Identity()

        logits = [linear(layer_nm(d(hidden_states))) for d in self.dropouts]

        losses = [loss_fn(log[mask].view(-1), labels.view(-1)) for log in logits]

        logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        loss = torch.mean(torch.stack(losses, dim=0), dim=0)

        return (loss, logits)