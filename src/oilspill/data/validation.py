"""Dataset audit, integrity validation, and class distribution analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import cv2
import numpy as np
import pandas as pd

from .dataset import CLASS_NAMES, rgb_to_mask
from ..utils.logging import get_logger

logger = get_logger("DatasetAuditor")


class DatasetAuditor:
    """Audits SAR dataset integrity, dimensions, label values, and class distributions."""

    def __init__(self, raw_dir: Union[str, Path]):
        self.raw_dir = Path(raw_dir)

    def scan_pairs(
        self,
        images_dir: Optional[Union[str, Path]] = None,
        masks_dir: Optional[Union[str, Path]] = None,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """Scan directory and pair images with their corresponding masks by stem name.
        
        Returns:
            matched_images: List of image paths.
            matched_masks: List of matching mask paths.
            unmatched: List of filenames missing pair.
        """
        img_folder = Path(images_dir) if images_dir else self.raw_dir / "images"
        msk_folder = Path(masks_dir) if masks_dir else self.raw_dir / "labels"
        
        if not img_folder.exists() and self.raw_dir.exists():
            # Alternative layout: check subdirectories or direct folders
            if (self.raw_dir / "train" / "images").exists():
                img_folder = self.raw_dir / "train" / "images"
                msk_folder = self.raw_dir / "train" / "labels"
            elif (self.raw_dir / "all_images").exists():
                img_folder = self.raw_dir / "all_images"
                msk_folder = self.raw_dir / "all_masks"

        if not img_folder.exists():
            # If no subfolder, scan raw_dir directly
            img_files = [
                p for p in self.raw_dir.rglob("*")
                if p.suffix.lower() in valid_extensions and "mask" not in p.stem.lower() and "label" not in p.stem.lower()
            ]
            msk_files = [
                p for p in self.raw_dir.rglob("*")
                if p.suffix.lower() in valid_extensions and ("mask" in p.stem.lower() or "label" in p.stem.lower())
            ]
        else:
            img_files = [p for p in img_folder.glob("*") if p.suffix.lower() in valid_extensions]
            msk_files = [p for p in msk_folder.glob("*") if p.suffix.lower() in valid_extensions] if msk_folder.exists() else []

        img_map = {p.stem: p for p in img_files}
        msk_map = {p.stem: p for p in msk_files}

        # Normalize stems if masks have suffixes like _mask or _label
        cleaned_msk_map = {}
        for stem, path in msk_map.items():
            clean_stem = stem.replace("_mask", "").replace("_label", "").replace("_mask_1ch", "").replace("_visual", "")
            cleaned_msk_map[clean_stem] = path

        matched_images = []
        matched_masks = []
        unmatched = []

        for stem, img_p in img_map.items():
            if stem in cleaned_msk_map:
                matched_images.append(img_p)
                matched_masks.append(cleaned_msk_map[stem])
            elif stem in msk_map:
                matched_images.append(img_p)
                matched_masks.append(msk_map[stem])
            else:
                unmatched.append(img_p.name)

        return sorted(matched_images), sorted(matched_masks), unmatched

    def audit(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform a comprehensive audit of images and masks."""
        total_pairs = len(image_paths)
        if max_samples and max_samples < total_pairs:
            sample_indices = np.random.choice(total_pairs, max_samples, replace=False)
            img_subset = [image_paths[i] for i in sample_indices]
            msk_subset = [mask_paths[i] for i in sample_indices]
        else:
            img_subset = image_paths
            msk_subset = mask_paths

        class_pixel_counts = {i: 0 for i in range(5)}
        class_image_presence = {i: 0 for i in range(5)}
        dimension_set: Set[Tuple[int, int]] = set()
        invalid_label_pixels = 0
        corrupted_files = []

        total_pixels = 0

        for img_p, msk_p in zip(img_subset, msk_subset):
            try:
                img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    corrupted_files.append(str(img_p))
                    continue
                
                msk_raw = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
                if msk_raw is None:
                    corrupted_files.append(str(msk_p))
                    continue

                if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
                    msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
                else:
                    msk = msk_raw

                dimension_set.add((img.shape[0], img.shape[1]))
                
                # Verify dimensions match between image and mask
                if img.shape[:2] != msk.shape[:2]:
                    corrupted_files.append(f"Dimension mismatch: {img_p.name} ({img.shape}) vs {msk_p.name} ({msk.shape})")
                    continue

                # Check unique values
                uniques, counts = np.unique(msk, return_counts=True)
                present_classes = set()
                for u, count in zip(uniques, counts):
                    if 0 <= u <= 4:
                        class_pixel_counts[int(u)] += int(count)
                        present_classes.add(int(u))
                    else:
                        invalid_label_pixels += int(count)
                
                for c in present_classes:
                    class_image_presence[c] += 1
                
                total_pixels += msk.size

            except Exception as e:
                corrupted_files.append(f"{img_p.name}: {str(e)}")

        # Compute percentages and class weights
        class_percentages = {}
        class_weights = {}
        total_valid_pixels = sum(class_pixel_counts.values())

        if total_valid_pixels > 0:
            for c, cnt in class_pixel_counts.items():
                class_percentages[CLASS_NAMES[c]] = float((cnt / total_valid_pixels) * 100.0)
                # Compute inverse frequency weight: median_freq / freq
                freq = cnt / total_valid_pixels
                class_weights[c] = float(freq)

            # Compute normalized balanced class weights (Median Frequency Balancing)
            freqs = np.array([class_pixel_counts[c] / total_valid_pixels for c in range(5)])
            # Avoid division by zero
            safe_freqs = np.where(freqs > 0, freqs, 1e-6)
            median_freq = np.median(safe_freqs[freqs > 0]) if np.any(freqs > 0) else 1.0
            balanced_weights = np.clip(median_freq / safe_freqs, 0.1, 50.0)
            balanced_weights_dict = {CLASS_NAMES[i]: float(balanced_weights[i]) for i in range(5)}
        else:
            balanced_weights_dict = {CLASS_NAMES[i]: 1.0 for i in range(5)}

        report = {
            "total_pairs_analyzed": len(img_subset),
            "dimensions_found": [list(d) for d in sorted(dimension_set)],
            "corrupted_or_mismatched_files": corrupted_files,
            "invalid_label_pixels": invalid_label_pixels,
            "total_pixels": total_valid_pixels,
            "class_pixel_counts": {CLASS_NAMES[k]: v for k, v in class_pixel_counts.items()},
            "class_pixel_percentages": class_percentages,
            "class_image_presence_count": {CLASS_NAMES[k]: v for k, v in class_image_presence.items()},
            "class_image_presence_percentage": {
                CLASS_NAMES[k]: float((v / len(img_subset) * 100.0)) if img_subset else 0.0
                for k, v in class_image_presence.items()
            },
            "suggested_class_weights": balanced_weights_dict,
        }

        return report

    def generate_markdown_report(self, audit_dict: Dict[str, Any]) -> str:
        """Format audit dictionary into a clean GitHub Flavored Markdown report."""
        md = []
        md.append("# 🛰️ SAR Oil Spill Dataset Audit & Integrity Report\n")
        md.append(f"**Total Image/Mask Pairs Audited:** {audit_dict['total_pairs_analyzed']}\n")
        md.append(f"**Image Dimensions Encountered (H × W):** {audit_dict['dimensions_found']}\n")
        md.append(f"**Invalid Label Pixels Detected:** {audit_dict['invalid_label_pixels']}\n")
        md.append(f"**Corrupted / Mismatched Files:** {len(audit_dict['corrupted_or_mismatched_files'])}\n")
        
        md.append("## 📊 Class Pixel Distribution & Imbalance Analysis\n")
        md.append("| Class ID | Class Name | Total Pixels | Pixel % | Image Presence % | Suggested Loss Weight |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for i in range(5):
            c_name = CLASS_NAMES[i]
            count = audit_dict['class_pixel_counts'].get(c_name, 0)
            pct = audit_dict['class_pixel_percentages'].get(c_name, 0.0)
            pres = audit_dict['class_image_presence_percentage'].get(c_name, 0.0)
            wt = audit_dict['suggested_class_weights'].get(c_name, 1.0)
            md.append(f"| {i} | `{c_name}` | {count:,} | {pct:.4f}% | {pres:.2f}% | {wt:.2f} |")

        md.append("\n## 🔬 Key Scientific Observations\n")
        md.append("- **Class Imbalance:** Background `sea_surface` dominates the pixel share. Pure pixel accuracy is insufficient.")
        md.append("- **Target Class:** `oil_spill` constitutes a small percentage of total pixels, highlighting the need for Weighted Cross-Entropy + Dice Loss.")
        md.append("- **Ambiguity Diagnostic:** `look_alike` regions represent confounding non-oil dark phenomena; careful spatial modeling and false-positive tracking are required.")

        return "\n".join(md)


def audit_dataset_directory(
    raw_dir: Union[str, Path],
    output_report_json: Optional[Union[str, Path]] = None,
    output_report_md: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Audit dataset directory and save reports."""
    auditor = DatasetAuditor(raw_dir)
    images, masks, unmatched = auditor.scan_pairs()
    
    if not images:
        logger.warning(f"No paired image/mask files found in {raw_dir}")
        return {"error": "No pairs found", "unmatched": unmatched}
        
    report = auditor.audit(images, masks)
    
    if output_report_json:
        out_json = Path(output_report_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
    if output_report_md:
        out_md = Path(output_report_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = auditor.generate_markdown_report(report)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
            
    return report
