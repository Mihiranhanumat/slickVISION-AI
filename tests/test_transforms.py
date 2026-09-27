"""Unit tests for augmentations and normalization."""

import numpy as np
import pytest

from oilspill.data.transforms import (
    get_training_transforms,
    get_validation_transforms,
    normalize_sar,
)


def test_training_transforms_preserves_discrete_labels():
    """Verify augmentations preserve discrete integer mask labels in {0, 1, 2, 3, 4}."""
    transforms = get_training_transforms()
    if transforms is None:
        pytest.skip("Albumentations not installed.")

    # Create dummy 256x256 image and mask
    img = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    mask = np.random.randint(0, 5, size=(256, 256), dtype=np.uint8)

    # Run augmentations multiple times
    for _ in range(10):
        augmented = transforms(image=img, mask=mask)
        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        assert aug_img.shape == (256, 256)
        assert aug_mask.shape == (256, 256)
        # Ensure mask contains only integer class IDs in {0, 1, 2, 3, 4}
        unique_labels = np.unique(aug_mask)
        assert set(unique_labels).issubset({0, 1, 2, 3, 4}), f"Interpolation artifact found: {unique_labels}"


def test_normalize_sar():
    """Verify SAR normalization range and properties."""
    img = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    norm = normalize_sar(img)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
