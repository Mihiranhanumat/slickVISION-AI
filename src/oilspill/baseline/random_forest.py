"""Classical Machine Learning Baseline: Random Forest on engineered SAR pixel features."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from ..data.dataset import CLASS_NAMES, mask_to_rgb, rgb_to_mask
from ..evaluation.metrics import SegmentationMetrics
from ..utils.logging import get_logger

logger = get_logger("RandomForestBaseline")


class SARPixelFeatureExtractor:
    """Engineers multi-scale local texture, intensity, and gradient features from raw SAR imagery."""

    def __init__(
        self,
        window_sizes: Tuple[int, ...] = (3, 7, 15),
        include_gradients: bool = True,
        include_laplacian: bool = True,
    ):
        self.window_sizes = window_sizes
        self.include_gradients = include_gradients
        self.include_laplacian = include_laplacian

    @property
    def feature_dim(self) -> int:
        """Calculate total number of extracted features per pixel."""
        dim = 1  # Raw intensity
        dim += len(self.window_sizes) * 2  # Mean and std for each window
        if self.include_gradients:
            dim += 3  # Sobel dx, dy, gradient magnitude
        if self.include_laplacian:
            dim += 1  # Laplacian
        return dim

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract dense pixel feature matrix.
        
        Args:
            image: (H, W) SAR image (uint8 or float).
            
        Returns:
            (H, W, D) feature cube where D is feature dimension.
        """
        img_f = image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        features = [img_f]

        # Multi-scale local statistics (Mean and Standard Deviation)
        for w in self.window_sizes:
            ksize = (w, w)
            mean = cv2.blur(img_f, ksize)
            mean_sq = cv2.blur(img_f ** 2, ksize)
            variance = np.maximum(mean_sq - (mean ** 2), 0.0)
            std = np.sqrt(variance)
            features.extend([mean, std])

        # Multi-scale gradients and directional responses
        if self.include_gradients:
            dx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(dx ** 2 + dy ** 2)
            features.extend([dx, dy, grad_mag])

        if self.include_laplacian:
            lap = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
            features.append(lap)

        feature_cube = np.stack(features, axis=-1)  # (H, W, D)
        return feature_cube


class RandomForestBaseline:
    """Random Forest semantic segmentation baseline."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 10,
        n_jobs: int = -1,
        random_state: int = 42,
        window_sizes: Tuple[int, ...] = (3, 7, 15),
    ):
        self.feature_extractor = SARPixelFeatureExtractor(window_sizes=window_sizes)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        self.is_fitted = False

    def sample_training_pixels(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        samples_per_class: int = 30000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract balanced/stratified pixel samples across training scenes.
        
        Args:
            image_paths: List of training SAR image paths.
            mask_paths: List of training mask paths.
            samples_per_class: Max samples per class across all training scenes.
            random_state: Seed for random sampling.
            
        Returns:
            X: (N, D) feature matrix.
            y: (N,) target labels.
        """
        rng = np.random.default_rng(random_state)
        class_pixels: Dict[int, List[np.ndarray]] = {c: [] for c in range(5)}

        logger.info(f"Extracting features from {len(image_paths)} training images...")

        for img_p, msk_p in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
            if msk_raw is None:
                continue

            if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
                msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
            else:
                msk = np.clip(msk_raw, 0, 4).astype(np.uint8)

            feats = self.feature_extractor.extract_features(img)  # (H, W, D)
            h, w, d = feats.shape
            flat_feats = feats.reshape(-1, d)
            flat_msk = msk.ravel()

            for c in range(5):
                c_indices = np.where(flat_msk == c)[0]
                if len(c_indices) > 0:
                    # Subsample per scene to manage memory
                    max_scene_samples = min(len(c_indices), max(200, samples_per_class // len(image_paths) * 3))
                    chosen = rng.choice(c_indices, size=max_scene_samples, replace=False)
                    class_pixels[c].append(flat_feats[chosen])

        # Concatenate and balance
        X_list = []
        y_list = []

        for c in range(5):
            if class_pixels[c]:
                c_all = np.vstack(class_pixels[c])
                if len(c_all) > samples_per_class:
                    c_all = rng.choice(c_all, size=samples_per_class, replace=False)
                X_list.append(c_all)
                y_list.append(np.full(len(c_all), c, dtype=np.uint8))
                logger.info(f"Class {c} ({CLASS_NAMES[c]}): {len(c_all):,} pixels sampled")
            else:
                logger.warning(f"Class {c} ({CLASS_NAMES[c]}) had 0 pixels in training images!")

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Shuffle
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def fit(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        samples_per_class: int = 30000,
    ) -> None:
        """Fit Random Forest model on training SAR data."""
        X, y = self.sample_training_pixels(
            image_paths=image_paths,
            mask_paths=mask_paths,
            samples_per_class=samples_per_class,
        )
        logger.info(f"Training Random Forest on {X.shape[0]:,} samples with {X.shape[1]} features...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest baseline training completed successfully.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict 5-class segmentation mask for a single SAR image.
        
        Args:
            image: (H, W) SAR image.
            
        Returns:
            (H, W) uint8 class prediction mask.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        h, w = image.shape[:2]
        feats = self.feature_extractor.extract_features(image)  # (H, W, D)
        flat_feats = feats.reshape(-1, feats.shape[-1])

        flat_pred = self.model.predict(flat_feats)
        return flat_pred.reshape(h, w).astype(np.uint8)

    def evaluate(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        output_pred_dir: Optional[Union[str, Path]] = None,
        max_visualize: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate baseline model against held-out validation or test images."""
        metrics = SegmentationMetrics(num_classes=5)
        
        if output_pred_dir:
            out_dir = Path(output_pred_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluating Random Forest on {len(image_paths)} held-out images...")

        saved_viz_count = 0
        for idx, (img_p, msk_p) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths))):
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
            if msk_raw is None:
                continue

            if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
                gt_mask = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
            else:
                gt_mask = np.clip(msk_raw, 0, 4).astype(np.uint8)

            pred_mask = self.predict(img)
            metrics.update(gt_mask, pred_mask)

            # Save qualitative visualization
            if output_pred_dir and saved_viz_count < max_visualize:
                stem = img_p.stem
                gt_rgb = mask_to_rgb(gt_mask)
                pred_rgb = mask_to_rgb(pred_mask)

                # SAR RGB version
                sar_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Oil overlay (Cyan on SAR)
                oil_overlay = sar_rgb.copy()
                oil_mask = (pred_mask == 1)
                oil_overlay[oil_mask] = [0, 255, 255]

                # Stack side-by-side: [SAR, Ground Truth, RF Prediction, Oil Overlay]
                comparison = np.hstack([sar_rgb, gt_rgb, pred_rgb, oil_overlay])
                viz_path = Path(output_pred_dir) / f"{stem}_rf_eval.png"
                cv2.imwrite(str(viz_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                saved_viz_count += 1

        summary = metrics.compute_summary()
        return summary

    def save(self, model_path: Union[str, Path]) -> None:
        """Save trained model to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))
        logger.info(f"Saved Random Forest baseline model to {path}")

    def load(self, model_path: Union[str, Path]) -> None:
        """Load trained model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(str(path))
        self.is_fitted = True
        logger.info(f"Loaded Random Forest model from {path}")
