"""Task losses (per-graph, for beneficial carriage) and dataset metrics (for the load check).

Two distinct roles:

  * ``per_graph_loss`` reduces one graph's [., T] prediction/target to a scalar loss, using
    the same reduction as the model's training loss (mean over T). Beneficial carriage
    decomposes L_clean - L_swap, so this must match cfg.model.loss_fun. It is applied to the
    clean graph (for dL/dh) and to every swapped replica (for L_swap).

  * ``dataset_metric`` recomputes the model's reported test metric (MAE or AP) over the full
    eval split -- the strongest checkpoint-load check. Direction (lower/higher better) and an
    abort threshold come from the GritTaskSpec.
"""

from __future__ import annotations

import numpy as np


def per_graph_loss(pred, y, loss_fun: str):
    """Per-row scalar loss reduced over the T targets (differentiable; keeps torch grads).

    pred, y: [., T] tensors. Returns a [.] tensor. Matches GRIT's cfg.model.loss_fun:
      l1               -> mean_t |pred - y|                 (regression, incl. multi-target)
      mse / l2         -> mean_t (pred - y)^2
      cross_entropy    -> mean_t BCEWithLogits(pred_t, y_t) (multilabel classification)
    """
    import torch.nn.functional as Fnn

    if pred.dim() == 1:
        pred = pred.view(1, -1)
    y = y.view(pred.shape).to(pred.dtype)
    lf = str(loss_fun).lower()
    if lf in ("l1", "mae"):
        return (pred - y).abs().mean(dim=-1)
    if lf in ("mse", "l2"):
        return (pred - y).pow(2).mean(dim=-1)
    if lf in ("cross_entropy", "bce", "binary_cross_entropy"):
        # GRIT uses BCEWithLogits for classification_multilabel targets.
        return Fnn.binary_cross_entropy_with_logits(pred, y, reduction="none").mean(dim=-1)
    raise ValueError(f"unsupported loss_fun for carriage: {loss_fun!r}")


def loss_units(loss_fun: str) -> str:
    lf = str(loss_fun).lower()
    if lf in ("l1", "mae"):
        return "MAE units"
    if lf in ("mse", "l2"):
        return "MSE units"
    if lf in ("cross_entropy", "bce", "binary_cross_entropy"):
        return "BCE nats"
    return "loss units"


def mae_metric(preds: np.ndarray, trues: np.ndarray) -> float:
    """Mean absolute error over all elements (regression, incl. multi-target)."""
    return float(np.abs(np.asarray(preds) - np.asarray(trues)).mean())


def multilabel_ap_metric(preds: np.ndarray, trues: np.ndarray) -> float:
    """Unweighted mean average-precision over labels (OGB peptides-func convention).

    preds are logits; average_precision_score is threshold-free, so no sigmoid is needed.
    Labels with no positive (or no negative) example are skipped, matching OGB's Evaluator.
    """
    from sklearn.metrics import average_precision_score

    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    aps = []
    for t in range(trues.shape[1]):
        yt = trues[:, t]
        if yt.min() == yt.max():  # only one class present -> AP undefined
            continue
        aps.append(average_precision_score(yt, preds[:, t]))
    return float(np.mean(aps)) if aps else float("nan")
