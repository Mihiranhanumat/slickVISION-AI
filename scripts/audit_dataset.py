"""Dataset audit and visual QA contact sheet generator."""

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from oilspill.data.dataset import CLASS_COLORS, CLASS_NAMES, mask_to_rgb, rgb_to_mask
from oilspill.data.validation import audit_dataset_directory
from oilspill.utils.io import load_yaml
from oilspill.utils.logging import get_logger

logger = get_logger("AuditAndQA")


def create_visual_qa_contact_sheet(
    image_paths: list,
    mask_paths: list,
    output_path: Path,
    num_samples: int = 6
) -> None:
    """Generate a high-quality visual QA contact sheet with SAR, Mask, Overlay, and Legend."""
    num_samples = min(num_samples, len(image_paths))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3.5 * num_samples), dpi=150)
    
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        img = cv2.imread(str(image_paths[i]), cv2.IMREAD_GRAYSCALE)
        msk_raw = cv2.imread(str(mask_paths[i]), cv2.IMREAD_UNCHANGED)
        
        if msk_raw.ndim == 3 and msk_raw.shape[2] >= 3:
            msk = rgb_to_mask(cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB))
        else:
            msk = np.clip(msk_raw, 0, 4).astype(np.uint8)

        msk_rgb = mask_to_rgb(msk)
        
        # Overlay
        sar_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(sar_rgb, 0.65, msk_rgb, 0.35, 0)

        # 1. SAR
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"SAR Scene: {image_paths[i].stem}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Mask
        axes[i, 1].imshow(msk_rgb)
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # 3. Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("SAR + Annotation Overlay", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    # Add unified legend
    legend_elements = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255.0, label=f"{c}: {CLASS_NAMES[c]}")
        for c in range(5)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=11,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Visual QA contact sheet to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset and produce statistics + visual QA.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--samples-qa", type=int, default=6, help="Number of samples on QA contact sheet")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = reports_dir / "dataset_audit_report.json"
    md_report_path = reports_dir / "dataset_audit_report.md"

    logger.info(f"Auditing dataset in {raw_dir}...")
    report = audit_dataset_directory(
        raw_dir=raw_dir,
        output_report_json=json_report_path,
        output_report_md=md_report_path,
    )

    # Visual QA
    from oilspill.data.validation import DatasetAuditor
    auditor = DatasetAuditor(raw_dir)
    images, masks, _ = auditor.scan_pairs()
    if images:
        qa_contact_sheet = reports_dir / "visual_qa_contact_sheet.png"
        create_visual_qa_contact_sheet(images, masks, qa_contact_sheet, num_samples=args.samples_qa)

    print("\n" + "=" * 80)
    print("[SUCCESS] AUDIT AND VISUAL QA COMPLETE")
    print("=" * 80)
    print(f"Detailed Report:   {md_report_path}")
    print(f"QA Contact Sheet:  {reports_dir / 'visual_qa_contact_sheet.png'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
