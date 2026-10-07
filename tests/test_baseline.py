"""Unit tests for SAR feature extractor and Random Forest baseline."""

import numpy as np
import pytest
from pathlib import Path
import cv2

from oilspill.baseline.random_forest import SARPixelFeatureExtractor, RandomForestBaseline


def test_sar_pixel_feature_extractor():
    """Verify feature extractor produces correct channels and finite values."""
    extractor = SARPixelFeatureExtractor(window_sizes=(3, 7), include_gradients=True, include_laplacian=True)
    
    # 1 intensity + 2*2 (mean+std) + 3 gradients + 1 laplacian = 9 features
    expected_dim = 1 + (2 * 2) + 3 + 1
    assert extractor.feature_dim == expected_dim

    img = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    feats = extractor.extract_features(img)

    assert feats.shape == (100, 100, expected_dim)
    assert np.all(np.isfinite(feats))


def test_random_forest_fit_predict_pipeline(tmp_path):
    """Verify Random Forest baseline trains and predicts on small synthetic data."""
    img_dir = tmp_path / "train_imgs"
    msk_dir = tmp_path / "train_msks"
    img_dir.mkdir()
    msk_dir.mkdir()

    img_paths = []
    msk_paths = []

    for i in range(2):
        img_p = img_dir / f"scene_{i}.png"
        msk_p = msk_dir / f"scene_{i}.png"

        img = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
        msk = np.zeros((100, 100), dtype=np.uint8)
        msk[20:40, 20:40] = 1 # Oil
        msk[60:80, 60:80] = 2 # Lookalike

        cv2.imwrite(str(img_p), img)
        cv2.imwrite(str(msk_p), msk)

        img_paths.append(img_p)
        msk_paths.append(msk_p)

    rf = RandomForestBaseline(n_estimators=10, max_depth=5, window_sizes=(3, 5))
    rf.fit(image_paths=img_paths, mask_paths=msk_paths, samples_per_class=500)

    assert rf.is_fitted

    test_img = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
    pred_mask = rf.predict(test_img)

    assert pred_mask.shape == (100, 100)
    assert set(np.unique(pred_mask)).issubset({0, 1, 2, 3, 4})
