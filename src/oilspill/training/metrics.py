"""Segmentation metrics computed from a confusion matrix (fast, no per-pixel Python loops)."""

from typing import Dict, List, Optional

import numpy as np
import torch

BINARY_NAMES = ["background", "oil_spill"]


def confusion_from_tensors(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Rows = ground truth, columns = prediction. Works on any device."""
    p = preds.reshape(-1).long()
    t = targets.reshape(-1).long()
    valid = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    idx = t[valid] * num_classes + p[valid]
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _nan_to_none(x: float) -> Optional[float]:
    return None if (x is None or np.isnan(x)) else float(x)


def metrics_from_confusion(cm, class_names: Optional[List[str]] = None) -> Dict:
    """Per-class and mean IoU / Dice / precision / recall plus flat 'oil_*' keys (class 1)."""
    cm = np.asarray(cm, dtype=np.float64)
    c = cm.shape[0]
    names = class_names or (BINARY_NAMES if c == 2 else [f"class_{i}" for i in range(c)])
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    present = cm.sum(1) > 0  # classes that exist in the ground truth

    def mean(a):
        a = a[present & ~np.isnan(a)]
        return float(a.mean()) if a.size else 0.0

    out: Dict = {
        "per_class": {
            names[i]: {"iou": _nan_to_none(iou[i]), "dice": _nan_to_none(dice[i]),
                       "precision": _nan_to_none(prec[i]), "recall": _nan_to_none(rec[i]),
                       "support_pixels": int(cm[i].sum())}
            for i in range(c)
        },
        "mean_iou": mean(iou), "mean_dice": mean(dice),
        "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0,
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }
    if c >= 2:  # class 1 is the oil-spill class
        g = lambda a: float(np.nan_to_num(a[1]))
        out.update(oil_iou=g(iou), oil_dice=g(dice), oil_f1=g(dice), oil_precision=g(prec), oil_recall=g(rec))
    return out


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
                    class_names: Optional[List[str]] = None) -> Dict:
    """Convenience wrapper: integer prediction / target tensors -> metrics dict."""
    cm = confusion_from_tensors(preds, targets, num_classes).cpu().numpy()
    return metrics_from_confusion(cm, class_names)


def detection_metrics(y_true, y_pred) -> Dict[str, float]:
    """Image-level oil detection (is there oil in this scene?) from boolean arrays."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(len(y_true), 1)}
