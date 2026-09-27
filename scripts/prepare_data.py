"""Single-command data pipeline: Audit, Deterministic Split, Patch Generation, and Manifest Builder."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from oilspill.data.dataset import CLASS_NAMES, rgb_to_mask
from oilspill.data.patching import extract_dataset_patches
from oilspill.data.validation import DatasetAuditor
from oilspill.utils.io import load_yaml, save_json
from oilspill.utils.logging import get_logger
from oilspill.utils.seed import set_seed

logger = get_logger("PrepareData")


def load_scene_pairs(folder: Path) -> Tuple[List[Path], List[Path]]:
    """Scan directory for paired images and labels."""
    auditor = DatasetAuditor(folder)
    images, masks, unmatched = auditor.scan_pairs()
    if unmatched:
        logger.warning(f"Unmatched files in {folder}: {unmatched[:5]} (total {len(unmatched)})")
    return images, masks


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset: split, extract patches, compute statistics.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base configuration YAML")
    parser.add_argument("--auto-generate", action="store_true", help="Auto-generate synthetic data if raw is empty")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    patches_dir = Path(cfg["paths"]["patches_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    
    for d in [patches_dir, splits_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if raw data exists with image files
    img_files = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.bmp"))
    if not raw_dir.exists() or len(img_files) == 0:
        if args.auto_generate:
            logger.info("Raw data directory contains no image files. Auto-generating synthetic benchmark scenes...")
            from generate_synthetic_data import generate_benchmark
            generate_benchmark(output_dir=raw_dir, num_train=12, num_test=4, seed=seed)
        else:
            logger.error(
                f"Raw data directory '{raw_dir}' contains no image files.\n"
                "Please place Krestenitis dataset in data/raw/ or run with --auto-generate / python scripts/download_data.py --synthetic"
            )
            sys.exit(1)

    # 1. Discover scenes
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"

    if train_dir.exists() and test_dir.exists():
        train_raw_imgs, train_raw_msks = load_scene_pairs(train_dir)
        test_imgs, test_msks = load_scene_pairs(test_dir)
    else:
        all_imgs, all_msks = load_scene_pairs(raw_dir)
        if len(all_imgs) == 0:
            logger.error(f"No valid image/mask pairs found in {raw_dir}")
            sys.exit(1)
        # Default split 80% train, 20% test if flat directory
        train_raw_imgs, test_imgs, train_raw_msks, test_msks = train_test_split(
            all_imgs, all_msks, test_size=0.15, random_state=seed
        )

    logger.info(f"Loaded {len(train_raw_imgs)} official training scenes, {len(test_imgs)} official test scenes.")

    # 2. Strict Parent-Level Train/Validation Split
    val_ratio = cfg["dataset"].get("val_split_ratio", 0.20)
    train_imgs, val_imgs, train_msks, val_msks = train_test_split(
        train_raw_imgs, train_raw_msks, test_size=val_ratio, random_state=seed
    )

    logger.info(f"Split parent training scenes -> Train: {len(train_imgs)}, Validation: {len(val_imgs)}")
    logger.info(f"Test scenes (untouched): {len(test_imgs)}")

    # Save parent scene ID lists
    with open(splits_dir / "train_images.txt", "w") as f:
        f.write("\n".join([p.stem for p in train_imgs]))
    with open(splits_dir / "val_images.txt", "w") as f:
        f.write("\n".join([p.stem for p in val_imgs]))
    with open(splits_dir / "test_images.txt", "w") as f:
        f.write("\n".join([p.stem for p in test_imgs]))

    # 3. Audit Dataset
    auditor = DatasetAuditor(raw_dir)
    all_audit_imgs = train_imgs + val_imgs + test_imgs
    all_audit_msks = train_msks + val_msks + test_msks
    audit_results = auditor.audit(all_audit_imgs, all_audit_msks)
    
    audit_json_path = reports_dir / "dataset_audit_report.json"
    audit_md_path = reports_dir / "dataset_audit_report.md"
    save_json(audit_results, audit_json_path)
    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write(auditor.generate_markdown_report(audit_results))
    logger.info(f"Saved dataset audit report to {audit_json_path} and {audit_md_path}")

    # 4. Compute Training Normalization Statistics & Class Weights
    train_pixels_sum = 0.0
    train_pixels_sq_sum = 0.0
    train_total_pixels = 0
    train_class_counts = {c: 0 for c in range(5)}

    for img_p, msk_p in zip(train_imgs, train_msks):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
        if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
            msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
        else:
            msk = np.clip(msk_raw, 0, 4).astype(np.uint8)

        img_f = img.astype(np.float32) / 255.0
        train_pixels_sum += float(np.sum(img_f))
        train_pixels_sq_sum += float(np.sum(img_f ** 2))
        train_total_pixels += img_f.size
        
        counts = np.bincount(msk.ravel(), minlength=5)[:5]
        for c in range(5):
            train_class_counts[c] += int(counts[c])

    mean_sar = train_pixels_sum / train_total_pixels if train_total_pixels > 0 else 0.5
    var_sar = (train_pixels_sq_sum / train_total_pixels) - (mean_sar ** 2) if train_total_pixels > 0 else 0.06
    std_sar = float(np.sqrt(max(var_sar, 1e-6)))

    norm_stats = {
        "sar_mean": float(mean_sar),
        "sar_std": float(std_sar),
        "total_train_pixels": int(train_total_pixels),
    }
    save_json(norm_stats, splits_dir / "norm_stats.json")
    logger.info(f"Computed training normalization stats: mean={mean_sar:.4f}, std={std_sar:.4f}")

    # Compute training class weights
    total_valid_tr = sum(train_class_counts.values())
    freqs = np.array([train_class_counts[c] / total_valid_tr for c in range(5)])
    safe_freqs = np.where(freqs > 0, freqs, 1e-6)
    med_freq = np.median(safe_freqs[freqs > 0]) if np.any(freqs > 0) else 1.0
    class_weights = np.clip(med_freq / safe_freqs, 0.1, 50.0)
    class_weights_dict = {CLASS_NAMES[i]: float(class_weights[i]) for i in range(5)}
    save_json(class_weights_dict, splits_dir / "class_weights.json")

    # 5. Extract 256x256 Patches
    patch_size = cfg["patching"].get("patch_size", 256)
    train_stride = cfg["patching"].get("train_stride", 192)
    val_stride = cfg["patching"].get("val_stride", 256)
    test_stride = cfg["patching"].get("test_stride", 256)

    all_images = train_imgs + val_imgs + test_imgs
    all_masks = train_msks + val_msks + test_msks
    all_splits = (["train"] * len(train_imgs)) + (["val"] * len(val_imgs)) + (["test"] * len(test_imgs))

    df_manifest = extract_dataset_patches(
        image_paths=all_images,
        mask_paths=all_masks,
        splits=all_splits,
        output_dir=patches_dir,
        manifest_dir=splits_dir,
        patch_size=patch_size,
        train_stride=train_stride,
        val_stride=val_stride,
        test_stride=test_stride,
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Patches Directory:  {patches_dir}")
    print(f"Patch Manifest:     {splits_dir / 'patch_manifest.csv'}")
    print(f"Audit Report:       {audit_md_path}")
    print(f"Total Patches:      {len(df_manifest):,} (Train: {len(df_manifest[df_manifest['split']=='train'])}, Val: {len(df_manifest[df_manifest['split']=='val'])}, Test: {len(df_manifest[df_manifest['split']=='test'])})")
    print(f"Oil Patches:        {int(df_manifest['has_oil'].sum()):,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
