"""Augmentation and preprocessing transformations using Albumentations."""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    from albumentations.pytorch import ToTensorV2
    HAS_TORCH_ALBUMENTATIONS = True
except ImportError:
    HAS_TORCH_ALBUMENTATIONS = False


def normalize_sar(
    image: np.ndarray,
    mean: float = 0.5,
    std: float = 0.25,
    to_float: bool = True
) -> np.ndarray:
    """Normalize SAR image intensity safely without test-set leakage.
    
    Args:
        image: uint8 [0, 255] or float SAR image.
        mean: Training set mean intensity.
        std: Training set standard deviation.
        to_float: If True, returns float32 array in range [0, 1] or standardized.
        
    Returns:
        Normalized float32 numpy array.
    """
    img_f = image.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    if not to_float:
        return (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
    # Standardize or range scale
    return img_f


def get_training_transforms(
    image_size: Tuple[int, int] = (256, 256),
    horizontal_flip: float = 0.5,
    vertical_flip: float = 0.5,
    random_rotate90: float = 0.5,
    brightness_contrast: float = 0.3,
    to_tensor: bool = False
):
    """Build Albumentations augmentation pipeline for training patches.
    
    Ensures that mask labels are preserved through NEAREST neighbor interpolation.
    """
    if not HAS_ALBUMENTATIONS:
        return None
        
    transform_list = [
        A.HorizontalFlip(p=horizontal_flip),
        A.VerticalFlip(p=vertical_flip),
        A.RandomRotate90(p=random_rotate90),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=brightness_contrast
        ),
    ]
    
    if to_tensor:
        try:
            from albumentations.pytorch import ToTensorV2
            transform_list.append(ToTensorV2())
        except ImportError:
            pass
            
    return A.Compose(
        transform_list,
        is_check_shapes=False,
    )


def get_validation_transforms(to_tensor: bool = False):
    """Build Albumentations pipeline for validation/testing (deterministic, no spatial jitter)."""
    if not HAS_ALBUMENTATIONS:
        return None
        
    transform_list = []
    if to_tensor:
        try:
            from albumentations.pytorch import ToTensorV2
            transform_list.append(ToTensorV2())
        except ImportError:
            pass
            
    return A.Compose(transform_list, is_check_shapes=False)
