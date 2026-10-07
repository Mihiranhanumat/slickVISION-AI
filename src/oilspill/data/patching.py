"""256x256 patch extraction engine and manifest builder for SAR imagery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import rgb_to_mask
from ..utils.logging import get_logger

logger = get_logger("PatchExtractor")


class PatchExtractor:
    """Extracts patches of fixed size from high-resolution SAR scenes with configurable overlap."""

    def __init__(
        self,
        patch_size: int = 256,
        train_stride: int = 192,
        val_stride: int = 256,
        test_stride: int = 256,
    ):
        self.patch_size = patch_size
        self.train_stride = train_stride
        self.val_stride = val_stride
        self.test_stride = test_stride

    def compute_grid_coordinates(
        self,
        height: int,
        width: int,
        stride: int
    ) -> List[Tuple[int, int]]:
        """Calculate top-left (y, x) bounding box coordinates ensuring full image coverage.
        
        Args:
            height: Image height.
            width: Image width.
            stride: Stride in pixels between consecutive patches.
            
        Returns:
            List of (y_top, x_left) tuples.
        """
        y_coords = list(range(0, height - self.patch_size + 1, stride))
        if len(y_coords) == 0 or y_coords[-1] + self.patch_size < height:
            y_coords.append(height - self.patch_size)

        x_coords = list(range(0, width - self.patch_size + 1, stride))
        if len(x_coords) == 0 or x_coords[-1] + self.patch_size < width:
            x_coords.append(width - self.patch_size)

        # Remove potential duplicates while preserving order
        y_coords = sorted(list(set(y_coords)))
        x_coords = sorted(list(set(x_coords)))

        coords = []
        for y in y_coords:
            for x in x_coords:
                coords.append((y, x))

        return coords

    def extract_scene_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        parent_id: str,
        split: str,
        output_patch_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Extract all patches from a single SAR scene and write to disk.
        
        Returns:
            List of patch metadata dictionaries.
        """
        stride = self.train_stride if split == "train" else (
            self.val_stride if split == "val" else self.test_stride
        )
        h, w = image.shape[:2]
        coords = self.compute_grid_coordinates(h, w, stride)

        split_img_dir = output_patch_dir / split / "images"
        split_msk_dir = output_patch_dir / split / "masks"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_msk_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for idx, (y, x) in enumerate(coords):
            patch_id = f"{parent_id}_p{idx:04d}_y{y}_x{x}"
            
            img_patch = image[y : y + self.patch_size, x : x + self.patch_size]
            msk_patch = mask[y : y + self.patch_size, x : x + self.patch_size]

            # Save patches
            img_filename = f"{patch_id}.png"
            msk_filename = f"{patch_id}.png"
            
            img_out_path = split_img_dir / img_filename
            msk_out_path = split_msk_dir / msk_filename
            
            cv2.imwrite(str(img_out_path), img_patch)
            cv2.imwrite(str(msk_out_path), msk_patch)

            # Compute class breakdown
            counts = np.bincount(msk_patch.ravel(), minlength=5)[:5]
            oil_pixels = int(counts[1])
            has_oil = bool(oil_pixels > 0)

            records.append({
                "patch_id": patch_id,
                "parent_id": parent_id,
                "split": split,
                "y_top": y,
                "x_left": x,
                "patch_size": self.patch_size,
                "image_path": str(img_out_path.as_posix()),
                "mask_path": str(msk_out_path.as_posix()),
                "has_oil": has_oil,
                "oil_pixels": oil_pixels,
                "lookalike_pixels": int(counts[2]),
                "ship_pixels": int(counts[3]),
                "land_pixels": int(counts[4]),
                "sea_pixels": int(counts[0]),
                "total_pixels": int(self.patch_size * self.patch_size),
            })

        return records


def extract_dataset_patches(
    image_paths: List[Path],
    mask_paths: List[Path],
    splits: List[str],
    output_dir: Union[str, Path],
    manifest_dir: Union[str, Path],
    patch_size: int = 256,
    train_stride: int = 192,
    val_stride: int = 256,
    test_stride: int = 256,
) -> pd.DataFrame:
    """Extract patches across the entire dataset according to assigned splits.
    
    Args:
        image_paths: List of full image paths.
        mask_paths: List of full mask paths.
        splits: List of split labels ('train', 'val', 'test') for each image.
        output_dir: Destination folder for patches.
        manifest_dir: Destination folder for CSV manifest files.
        patch_size: Size of square patch.
        train_stride: Stride for training images.
        val_stride: Stride for validation images.
        test_stride: Stride for test images.
        
    Returns:
        DataFrame containing complete patch manifest.
    """
    output_dir = Path(output_dir)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    extractor = PatchExtractor(
        patch_size=patch_size,
        train_stride=train_stride,
        val_stride=val_stride,
        test_stride=test_stride,
    )

    all_records = []
    logger.info(f"Extracting patches from {len(image_paths)} scenes...")

    for img_p, msk_p, split in tqdm(zip(image_paths, mask_paths, splits), total=len(image_paths)):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image {img_p}")
            continue
            
        msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
        if msk_raw is None:
            logger.error(f"Failed to load mask {msk_p}")
            continue

        if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
            msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
        else:
            msk = np.clip(msk_raw, 0, 4).astype(np.uint8)

        parent_id = img_p.stem
        records = extractor.extract_scene_patches(
            image=img,
            mask=msk,
            parent_id=parent_id,
            split=split,
            output_patch_dir=output_dir,
        )
        all_records.extend(records)

    df_manifest = pd.DataFrame(all_records)
    
    # Save overall manifest and split manifests
    manifest_path = manifest_dir / "patch_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"Saved total patch manifest to {manifest_path} ({len(df_manifest)} patches)")

    for sp in ["train", "val", "test"]:
        sp_df = df_manifest[df_manifest["split"] == sp]
        if not sp_df.empty:
            sp_path = manifest_dir / f"{sp}_patches.csv"
            sp_df.to_csv(sp_path, index=False)
            oil_count = int(sp_df['has_oil'].sum())
            logger.info(f"Split '{sp}': {len(sp_df)} patches ({oil_count} with oil spills)")

    return df_manifest
