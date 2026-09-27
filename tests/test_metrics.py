"""Unit tests for semantic segmentation metrics, IoU, Dice, and confusion matrix."""

import numpy as np
import pytest

from oilspill.evaluation.metrics import (
    SegmentationMetrics,
    compute_confusion_matrix,
    compute_iou,
    compute_dice,
    compute_precision_recall,
)


def test_confusion_matrix_and_perfect_prediction():
    """Verify metrics on a perfect prediction match 1.0 exactly."""
    y_true = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    y_pred = np.array([0, 1, 2, 3, 4], dtype=np.uint8)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=5)
    assert np.array_equal(cm, np.eye(5, dtype=np.int64))

    iou = compute_iou(cm)
    assert np.allclose(iou, 1.0)

    dice = compute_dice(cm)
    assert np.allclose(dice, 1.0)


def test_iou_and_dice_analytical_values():
    """Verify IoU and Dice against hand-calculated analytical example.
    
    GT:   [1, 1, 1, 1, 0, 0]
    Pred: [1, 1, 0, 0, 1, 0]
    
    For Class 1 (Oil):
    TP = 2
    FP = 1
    FN = 2
    IoU = 2 / (2 + 1 + 2) = 2/5 = 0.40
    Dice = 2 * 2 / (2 * 2 + 1 + 2) = 4/7 ~ 0.5714
    """
    y_true = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    y_pred = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=5)
    iou = compute_iou(cm)
    dice = compute_dice(cm)

    assert pytest.approx(iou[1], 0.001) == 0.40
    assert pytest.approx(dice[1], 0.001) == 4.0 / 7.0


def test_segmentation_metrics_accumulator():
    """Verify SegmentationMetrics summary output structure."""
    engine = SegmentationMetrics(num_classes=5)
    
    mask_gt = np.zeros((100, 100), dtype=np.uint8)
    mask_gt[20:50, 20:50] = 1 # Oil spill
    mask_gt[60:80, 60:80] = 2 # Lookalike

    mask_pred = np.zeros((100, 100), dtype=np.uint8)
    mask_pred[20:50, 20:50] = 1 # Perfect oil
    mask_pred[60:70, 60:80] = 2 # Partial lookalike
    mask_pred[70:80, 60:80] = 1 # Confused lookalike as oil

    engine.update(mask_gt, mask_pred)
    summary = engine.compute_summary()

    assert "oil_iou" in summary
    assert "mean_iou" in summary
    assert "error_analysis" in summary
    assert summary["error_analysis"]["lookalike_confused_as_oil_pixels"] > 0
