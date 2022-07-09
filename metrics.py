from bisect import bisect
from itertools import chain

import pandas as pd
from sklearn.metrics import mean_absolute_error

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(predictions, ground_truth):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [
            gt.index(x) for x in pred
        ]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def ai4code_compute_metrics(eval_preds, eval_dataset):

    preds, bad_labels = eval_preds

    labels = eval_dataset["labels"]

    mask = bad_labels != -100

    preds = preds[mask].ravel()

    mae = mean_absolute_error(preds, list(chain(*labels)))

    pred_ids = []
    idx = 0
    for cell_ids, cell_types in zip(eval_dataset["cell_ids"], eval_dataset["cell_type"]):
        num2add = len(cell_ids)



        temp_df = pd.DataFrame(
            {
                "scores": preds[idx : idx + num2add],
                "cell_ids": cell_ids,
                "cell_type": cell_types
            }
        )
        temp_df.loc[temp_df.cell_type=="code", "scores"] = temp_df.loc[temp_df.cell_type=="code", "scores"].rank(pct=True)
        temp_df = temp_df.sort_values(by="scores")
        pred_ids.append(temp_df["cell_ids"].tolist())

        idx += num2add
        
    assert idx == len(preds)

    kt = kendall_tau(pred_ids, eval_dataset["correct_order"])

    return {
        "mae": mae,
        "kt": kt,
    }