"""Evaluation metrics, confusion matrix analysis, and validation utilities."""

from .metrics import (
    SegmentationMetrics,
    compute_iou,
    compute_dice,
    compute_precision_recall,
    evaluate_segmentation_batch,
)
from .confusion import (
    ConfusionMatrixAnalyzer,
    plot_confusion_matrix,
)

__all__ = [
    "SegmentationMetrics",
    "compute_iou",
    "compute_dice",
    "compute_precision_recall",
    "evaluate_segmentation_batch",
    "ConfusionMatrixAnalyzer",
    "plot_confusion_matrix",
]
