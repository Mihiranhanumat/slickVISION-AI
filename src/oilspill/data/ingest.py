"""Real SAR dataset ingestion engine for CSIRO / Kaggle Sentinel-1 dataset."""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from .dataset import mask_to_rgb
from ..utils.logging import get_logger

logger = get_logger("DatasetIngest")


def generate_chip_mask(
    image: np.ndarray,
    is_oil: bool,
    blur_ksize: int = 5
) -> np.ndarray:
    """Generate pixel-level semantic mask from real Sentinel-1 SAR chip.
    
    Args:
        image: (H, W) uint8 grayscale SAR image.
        is_oil: True if chip contains confirmed oil slick (Class_1), False for Class_0.
        blur_ksize: Gaussian blur kernel size for speckle suppression.
        
    Returns:
        (H, W) uint8 mask with class IDs (0: Sea/Non-oil, 1: Oil spill).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not is_oil:
        # Class 0: Clear sea or look-alike non-oil background
        return mask

    # Class 1: Oil spill - detect dampened capillary wave signatures
    blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    
    # Adaptive thresholding on dark backscatter regions
    # Otsu threshold finds the optimal boundary between background sea and dampened oil
    thresh_val, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Refine mask: remove isolated noise pixels using morphological closing & opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Mark oil spill pixels as Class 1
    mask[cleaned == 255] = 1
    
    # Safety fallback: if no pixels passed, take bottom 15% darkest pixels
    if np.sum(mask == 1) == 0:
        dark_thresh = np.percentile(blur, 15)
        mask[blur <= dark_thresh] = 1
        
    return mask


def ingest_kaggle_dataset(
    kaggle_data_dir: Path,
    output_raw_dir: Path,
    max_chips_per_class: Optional[int] = None
) -> Tuple[int, int]:
    """Ingest real CSIRO / Kaggle Sentinel-1 SAR dataset into data/raw pairing structure.
    
    Args:
        kaggle_data_dir: Path containing Class_0 and Class_1 subdirectories.
        output_raw_dir: Destination data/raw directory.
        max_chips_per_class: Optional subsampling limit (None for all).
        
    Returns:
        (total_class_0_ingested, total_class_1_ingested)
    """
    c0_dir = kaggle_data_dir / "Class_0"
    c1_dir = kaggle_data_dir / "Class_1"
    
    if not c0_dir.exists() or not c1_dir.exists():
        raise FileNotFoundError(f"Missing Class_0 or Class_1 directories in {kaggle_data_dir}")

    images_out = output_raw_dir / "images"
    labels_out = output_raw_dir / "labels"
    visual_out = output_raw_dir / "visual_masks"
    
    for d in [images_out, labels_out, visual_out]:
        d.mkdir(parents=True, exist_ok=True)

    c0_files = sorted(list(c0_dir.glob("*.jpg")) + list(c0_dir.glob("*.png")))
    c1_files = sorted(list(c1_dir.glob("*.jpg")) + list(c1_dir.glob("*.png")))

    if max_chips_per_class:
        c0_files = c0_files[:max_chips_per_class]
        c1_files = c1_files[:max_chips_per_class]

    logger.info(f"Ingesting real SAR dataset: {len(c0_files)} Class_0 (Non-oil) and {len(c1_files)} Class_1 (Oil) chips...")

    count_c0 = 0
    count_c1 = 0

    # Ingest Class 0 (Non-oil)
    for p in tqdm(c0_files, desc="Ingesting Class 0 (Non-oil)"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = generate_chip_mask(img, is_oil=False)
        stem = p.stem
        
        cv2.imwrite(str(images_out / f"{stem}.png"), img)
        cv2.imwrite(str(labels_out / f"{stem}.png"), mask)
        count_c0 += 1

    # Ingest Class 1 (Oil Spill)
    for p in tqdm(c1_files, desc="Ingesting Class 1 (Oil Spill)"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = generate_chip_mask(img, is_oil=True)
        stem = p.stem
        
        cv2.imwrite(str(images_out / f"{stem}.png"), img)
        cv2.imwrite(str(labels_out / f"{stem}.png"), mask)
        
        # Save visual overlay
        vis_mask = mask_to_rgb(mask)
        cv2.imwrite(str(visual_out / f"{stem}.png"), cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))
        count_c1 += 1

    logger.info(f"Successfully ingested {count_c0 + count_c1} real SAR scenes into {output_raw_dir}")
    return count_c0, count_c1
