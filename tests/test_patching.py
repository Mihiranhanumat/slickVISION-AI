"""Unit tests for patch extraction geometry, strides, and metadata manifest."""

import numpy as np
import pytest
from pathlib import Path
import cv2

from oilspill.data.patching import PatchExtractor, extract_dataset_patches


def test_grid_coordinates_coverage():
    """Verify compute_grid_coordinates covers entire image bounds without going out of bounds."""
    extractor = PatchExtractor(patch_size=256, train_stride=192, val_stride=256)
    
    # 650 x 1250 SAR scene
    h, w = 650, 1250
    coords = extractor.compute_grid_coordinates(h, w, stride=192)
    
    # Check bounds
    for y, x in coords:
        assert 0 <= y <= h - 256
        assert 0 <= x <= w - 256
        assert y + 256 <= h
        assert x + 256 <= w

    # Check that corners are reached
    max_y = max(y for y, x in coords)
    max_x = max(x for y, x in coords)
    assert max_y + 256 == h
    assert max_x + 256 == w


def test_patch_extraction_and_manifest(tmp_path):
    """Verify patch extraction generates correct files and valid manifest records."""
    img_dir = tmp_path / "raw_img"
    msk_dir = tmp_path / "raw_msk"
    patch_dir = tmp_path / "patches"
    manifest_dir = tmp_path / "splits"
    
    img_dir.mkdir()
    msk_dir.mkdir()
    
    img_p = img_dir / "scene_test.png"
    msk_p = msk_dir / "scene_test.png"
    
    # Dummy SAR scene
    img = np.full((650, 1250), 120, dtype=np.uint8)
    msk = np.zeros((650, 1250), dtype=np.uint8)
    # Add oil spill region
    msk[100:200, 100:200] = 1
    
    cv2.imwrite(str(img_p), img)
    cv2.imwrite(str(msk_p), msk)

    df_manifest = extract_dataset_patches(
        image_paths=[img_p],
        mask_paths=[msk_p],
        splits=["train"],
        output_dir=patch_dir,
        manifest_dir=manifest_dir,
        patch_size=256,
        train_stride=256,
        val_stride=256,
    )

    assert len(df_manifest) > 0
    assert "patch_id" in df_manifest.columns
    assert "has_oil" in df_manifest.columns
    assert "oil_pixels" in df_manifest.columns
    
    # Verify patch files exist and have correct shape (256, 256)
    first_patch_img = cv2.imread(df_manifest.iloc[0]["image_path"], cv2.IMREAD_GRAYSCALE)
    first_patch_msk = cv2.imread(df_manifest.iloc[0]["mask_path"], cv2.IMREAD_UNCHANGED)
    
    assert first_patch_img.shape == (256, 256)
    assert first_patch_msk.shape == (256, 256)
    assert df_manifest["has_oil"].sum() >= 1
