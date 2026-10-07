"""Dataset adapter for the Krestenitis et al. (2019) SAR Oil Spill Dataset."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

CLASS_NAMES = {
    0: "sea_surface",
    1: "oil_spill",
    2: "look_alike",
    3: "ship",
    4: "land",
}

# Standard official RGB color map for 5 classes
CLASS_COLORS = {
    0: (0, 0, 0),        # Sea surface - Black
    1: (0, 255, 255),    # Oil spill - Cyan
    2: (255, 0, 0),      # Look-alike - Red
    3: (150, 0, 150),    # Ship - Purple / Magenta
    4: (0, 100, 0),      # Land - Dark Green
}

# Mapping table for fast color -> class ID lookup
_RGB_PALETTE = np.array([
    [0, 0, 0],        # 0: Sea
    [0, 255, 255],    # 1: Oil spill (Cyan)
    [255, 0, 0],      # 2: Look-alike (Red)
    [150, 0, 150],    # 3: Ship (Magenta/Purple)
    [0, 100, 0],      # 4: Land (Green)
], dtype=np.uint8)


def rgb_to_mask(rgb_image: np.ndarray, tolerance: int = 40) -> np.ndarray:
    """Convert an RGB visual mask to a single-channel integer mask (0-4).
    
    Args:
        rgb_image: (H, W, 3) uint8 RGB array.
        tolerance: Euclidean distance tolerance for color matching.
        
    Returns:
        (H, W) uint8 array with class IDs in range [0, 4].
    """
    if rgb_image.ndim == 2:
        # Already single channel
        return np.clip(rgb_image, 0, 4).astype(np.uint8)
        
    h, w, c = rgb_image.shape
    flat_rgb = rgb_image.reshape(-1, 3).astype(np.int32)  # (N, 3)
    palette = _RGB_PALETTE.astype(np.int32)                # (5, 3)
    
    # Compute squared Euclidean distances: (N, 5)
    # ||p - c||^2 = ||p||^2 + ||c||^2 - 2 * p . c
    dists = (
        np.sum(flat_rgb ** 2, axis=1, keepdims=True)
        + np.sum(palette ** 2, axis=1)
        - 2 * np.dot(flat_rgb, palette.T)
    )
    
    min_indices = np.argmin(dists, axis=1)
    mask = min_indices.reshape(h, w).astype(np.uint8)
    return mask


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel integer mask (0-4) to a 3-channel RGB image for visualization.
    
    Args:
        mask: (H, W) array with integer class IDs [0-4].
        
    Returns:
        (H, W, 3) uint8 RGB image.
    """
    mask_clipped = np.clip(mask, 0, 4).astype(np.int64)
    rgb = _RGB_PALETTE[mask_clipped]
    return rgb.astype(np.uint8)


class KrestenitisDataset:
    """Dataset adapter for SAR Oil Spill imagery and 5-class semantic masks."""
    
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        mask_paths: List[Union[str, Path]],
        transforms: Optional[Callable] = None,
        is_rgb_mask: Optional[bool] = None,
        preload_into_memory: bool = False,
    ):
        """
        Args:
            image_paths: List of file paths to SAR images.
            mask_paths: List of file paths to corresponding masks.
            transforms: Optional albumentations transform pipeline.
            is_rgb_mask: If True, explicitly converts RGB masks. If None, auto-detects.
            preload_into_memory: If True, preloads dataset into RAM for fast access.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch in count: {len(image_paths)} images vs {len(mask_paths)} masks."
            )
            
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.transforms = transforms
        self.is_rgb_mask = is_rgb_mask
        self.preload = preload_into_memory
        
        self._cached_images: Optional[List[np.ndarray]] = None
        self._cached_masks: Optional[List[np.ndarray]] = None
        
        if self.preload:
            self._load_all()

    def _load_all(self) -> None:
        self._cached_images = []
        self._cached_masks = []
        for img_p, msk_p in zip(self.image_paths, self.mask_paths):
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image: {img_p}")
            msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
            if msk_raw is None:
                raise ValueError(f"Failed to read mask: {msk_p}")
            if msk_raw.ndim == 3:
                msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
            else:
                msk = np.clip(msk_raw, 0, 4).astype(np.uint8)
            self._cached_images.append(img)
            self._cached_masks.append(msk)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (image, mask) pair.
        
        Returns:
            image: (H, W) or (H, W, 1) float32/uint8 SAR array.
            mask: (H, W) uint8 array with class IDs {0, 1, 2, 3, 4}.
        """
        if self.preload and self._cached_images is not None:
            image = self._cached_images[idx].copy()
            mask = self._cached_masks[idx].copy()
        else:
            img_path = str(self.image_paths[idx])
            mask_path = str(self.mask_paths[idx])
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_raw is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
                
            if mask_raw.ndim == 3 and mask_raw.shape[2] >= 3:
                mask_rgb = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)
                mask = rgb_to_mask(mask_rgb)
            else:
                mask = np.clip(mask_raw, 0, 4).astype(np.uint8)
                
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
        return image, mask

    def get_info(self, idx: int) -> Dict[str, Union[str, int]]:
        """Retrieve metadata for a given item index."""
        return {
            "index": idx,
            "image_path": str(self.image_paths[idx]),
            "mask_path": str(self.mask_paths[idx]),
            "image_stem": self.image_paths[idx].stem,
        }
