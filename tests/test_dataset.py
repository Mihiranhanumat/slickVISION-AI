"""Unit tests for dataset adapter, RGB-to-mask conversion, and mask integrity."""

import numpy as np
import pytest
import cv2

from oilspill.data.dataset import (
    KrestenitisDataset,
    rgb_to_mask,
    mask_to_rgb,
    CLASS_NAMES,
    CLASS_COLORS,
)


def test_rgb_to_mask_exact_colors():
    """Verify exact RGB color mappings match expected class IDs."""
    h, w = 10, 10
    for cls_id, rgb in CLASS_COLORS.items():
        img = np.full((h, w, 3), rgb, dtype=np.uint8)
        mask = rgb_to_mask(img)
        assert mask.shape == (h, w)
        assert np.all(mask == cls_id), f"Failed mapping for class {cls_id} ({CLASS_NAMES[cls_id]})"


def test_rgb_to_mask_noisy_colors():
    """Verify nearest-color mapping handles mild compression artifacts."""
    h, w = 5, 5
    # Cyan [0, 255, 255] with noise -> [10, 245, 250]
    noisy_cyan = np.full((h, w, 3), [10, 245, 250], dtype=np.uint8)
    mask = rgb_to_mask(noisy_cyan)
    assert np.all(mask == 1), "Noisy cyan failed to map to Oil Spill (class 1)"

    # Red [255, 0, 0] with noise -> [240, 15, 10]
    noisy_red = np.full((h, w, 3), [240, 15, 10], dtype=np.uint8)
    mask_red = rgb_to_mask(noisy_red)
    assert np.all(mask_red == 2), "Noisy red failed to map to Look-alike (class 2)"


def test_mask_to_rgb_reversibility():
    """Verify roundtrip conversion mask -> rgb -> mask preserves labels."""
    original_mask = np.array([
        [0, 1, 2],
        [3, 4, 0],
        [1, 2, 3]
    ], dtype=np.uint8)
    
    rgb = mask_to_rgb(original_mask)
    assert rgb.shape == (3, 3, 3)
    
    recovered = rgb_to_mask(rgb)
    assert np.array_equal(original_mask, recovered)


def test_dataset_loader_and_shapes(tmp_path):
    """Test KrestenitisDataset indexing and loading."""
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    img_paths = []
    msk_paths = []
    
    for i in range(3):
        img_p = img_dir / f"scene_{i}.png"
        msk_p = msk_dir / f"scene_{i}.png"
        
        # Create 650x1250 dummy images
        fake_img = (np.random.rand(650, 1250) * 255).astype(np.uint8)
        fake_msk = np.random.randint(0, 5, size=(650, 1250), dtype=np.uint8)
        
        cv2.imwrite(str(img_p), fake_img)
        cv2.imwrite(str(msk_p), fake_msk)
        
        img_paths.append(img_p)
        msk_paths.append(msk_p)

    dataset = KrestenitisDataset(img_paths, msk_paths)
    assert len(dataset) == 3
    
    img, msk = dataset[0]
    assert img.shape == (650, 1250)
    assert msk.shape == (650, 1250)
    assert set(np.unique(msk)).issubset({0, 1, 2, 3, 4})
