---
{{ card_data }}
---

# {{ model_id | default("CoolModel") }}

This model is fine-tuned on the {{ dataset_name }} for the Kaggle AI4Code Competition.

## Metrics

```
{{ metrics }}
```

## Configuration

```
{{ config }}
```

## Weights and Biases

The run is logged at this url: https://wandb.ai/nbroad/ai4code/runs/{{ wandb_run_id }}