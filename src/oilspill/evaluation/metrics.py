"""Pixel-level and class-level semantic segmentation metrics calculation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..data.dataset import CLASS_NAMES


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> np.ndarray:
    """Compute confusion matrix where rows are ground truth and columns are predictions.
    
    Args:
        y_true: Ground truth integer array (N,) or (H, W).
        y_pred: Predicted integer array (N,) or (H, W).
        num_classes: Total number of classes (default 5).
        
    Returns:
        (num_classes, num_classes) numpy array of pixel counts.
    """
    y_t = y_true.ravel().astype(np.int64)
    y_p = y_pred.ravel().astype(np.int64)
    
    # Filter to valid range
    valid_mask = (y_t >= 0) & (y_t < num_classes) & (y_p >= 0) & (y_p < num_classes)
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]
    
    indices = y_t * num_classes + y_p
    cm = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def compute_iou(confusion_matrix: np.ndarray) -> np.ndarray:
    """Calculate per-class IoU (Jaccard Index) from confusion matrix.
    
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    denominator = tp + fp + fn
    
    iou = np.zeros(confusion_matrix.shape[0], dtype=np.float64)
    for c in range(len(tp)):
        if denominator[c] > 0:
            iou[c] = tp[c] / denominator[c]
        else:
            # If class neither appears in GT nor Pred, IoU is NaN or 1.0 (convention: NaN if absent)
            iou[c] = np.nan
    return iou


def compute_dice(confusion_matrix: np.ndarray) -> np.ndarray:
    """Calculate per-class Dice coefficient (F1 Score) from confusion matrix.
    
    Dice_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)
    """
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    denominator = 2 * tp + fp + fn
    
    dice = np.zeros(confusion_matrix.shape[0], dtype=np.float64)
    for c in range(len(tp)):
        if denominator[c] > 0:
            dice[c] = (2.0 * tp[c]) / denominator[c]
        else:
            dice[c] = np.nan
    return dice


def compute_precision_recall(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate per-class Precision and Recall from confusion matrix."""
    tp = np.diag(confusion_matrix)
    pred_sum = np.sum(confusion_matrix, axis=0)
    true_sum = np.sum(confusion_matrix, axis=1)
    
    precision = np.zeros(confusion_matrix.shape[0], dtype=np.float64)
    recall = np.zeros(confusion_matrix.shape[0], dtype=np.float64)
    
    for c in range(len(tp)):
        precision[c] = tp[c] / pred_sum[c] if pred_sum[c] > 0 else np.nan
        recall[c] = tp[c] / true_sum[c] if true_sum[c] > 0 else np.nan
        
    return precision, recall


class SegmentationMetrics:
    """Accumulator and metric summary engine for semantic segmentation."""

    def __init__(self, num_classes: int = 5, class_names: Optional[Dict[int, str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_images = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Accumulate predictions and ground truths."""
        cm = compute_confusion_matrix(y_true, y_pred, self.num_classes)
        self.confusion_matrix += cm
        self.total_images += 1

    def compute_summary(self) -> Dict[str, Any]:
        """Compute full metrics dictionary."""
        cm = self.confusion_matrix
        tp = np.diag(cm)
        total_pixels = np.sum(cm)
        overall_pixel_acc = float(np.sum(tp) / total_pixels) if total_pixels > 0 else 0.0

        iou = compute_iou(cm)
        dice = compute_dice(cm)
        precision, recall = compute_precision_recall(cm)

        # Mean metrics ignoring classes not present in test set
        valid_iou = iou[~np.isnan(iou)]
        mean_iou = float(np.mean(valid_iou)) if len(valid_iou) > 0 else 0.0
        
        valid_dice = dice[~np.isnan(dice)]
        mean_dice = float(np.mean(valid_dice)) if len(valid_dice) > 0 else 0.0

        per_class_metrics = {}
        for c in range(self.num_classes):
            name = self.class_names.get(c, f"class_{c}")
            per_class_metrics[name] = {
                "iou": float(iou[c]) if not np.isnan(iou[c]) else 0.0,
                "dice": float(dice[c]) if not np.isnan(dice[c]) else 0.0,
                "precision": float(precision[c]) if not np.isnan(precision[c]) else 0.0,
                "recall": float(recall[c]) if not np.isnan(recall[c]) else 0.0,
                "gt_pixels": int(np.sum(cm[c, :])),
                "pred_pixels": int(np.sum(cm[:, c])),
                "tp_pixels": int(tp[c]),
            }

        # Look-alike confusion diagnostics
        # Class 1 is Oil Spill, Class 2 is Look-alike
        oil_gt_total = np.sum(cm[1, :])
        lookalike_gt_total = np.sum(cm[2, :])
        
        oil_confused_as_lookalike = int(cm[1, 2])
        lookalike_confused_as_oil = int(cm[2, 1])
        
        oil_lookalike_leak_rate = (
            float(oil_confused_as_lookalike / oil_gt_total) if oil_gt_total > 0 else 0.0
        )
        lookalike_false_alarm_rate = (
            float(lookalike_confused_as_oil / lookalike_gt_total) if lookalike_gt_total > 0 else 0.0
        )

        return {
            "overall_pixel_accuracy": overall_pixel_acc,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "oil_iou": float(iou[1]) if not np.isnan(iou[1]) else 0.0,
            "oil_dice": float(dice[1]) if not np.isnan(dice[1]) else 0.0,
            "oil_precision": float(precision[1]) if not np.isnan(precision[1]) else 0.0,
            "oil_recall": float(recall[1]) if not np.isnan(recall[1]) else 0.0,
            "per_class": per_class_metrics,
            "error_analysis": {
                "oil_confused_as_lookalike_pixels": oil_confused_as_lookalike,
                "oil_to_lookalike_error_rate": oil_lookalike_leak_rate,
                "lookalike_confused_as_oil_pixels": lookalike_confused_as_oil,
                "lookalike_false_alarm_rate": lookalike_false_alarm_rate,
            },
            "confusion_matrix": cm.tolist(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-class summary to pandas DataFrame."""
        summary = self.compute_summary()
        records = []
        for name, m in summary["per_class"].items():
            records.append({
                "class": name,
                "IoU": m["iou"],
                "Dice_F1": m["dice"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "GT_Pixels": m["gt_pixels"],
                "Pred_Pixels": m["pred_pixels"],
            })
        df = pd.DataFrame(records)
        return df


def evaluate_segmentation_batch(
    y_trues: List[np.ndarray],
    y_preds: List[np.ndarray],
    num_classes: int = 5
) -> Dict[str, Any]:
    """Convenience helper to evaluate a list of true/predicted masks."""
    metric_engine = SegmentationMetrics(num_classes=num_classes)
    for yt, yp in zip(y_trues, y_preds):
        metric_engine.update(yt, yp)
    return metric_engine.compute_summary()
