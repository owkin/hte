"""Classification metrics for experiments."""
import numpy as np


def classification_metrics(true_group : np.array, pred_group : np.array):
    """Compute classification metrics (TP, TN, FP, FN)."""
    true0 = true_group == 0
    true1 = true_group == 1
    pred0 = pred_group == 0
    pred1 = pred_group == 1

    tp = (pred1 & true1).sum()
    tn = (pred0 & true0).sum()
    fp = (pred1 & true0).sum()
    fn = (pred0 & true1).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc, tp, tn, fp, fn
