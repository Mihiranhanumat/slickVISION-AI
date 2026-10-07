from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilspill.models.segformer_sar_5class import SegFormerSAR5Class
from train_segformer_5class import (
    CLASS_NAMES,
    FiveClassPatchDataset,
    compute_metrics,
    resolve_path,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    manifest = resolve_path("data/splits/patch_manifest.csv")
    test_ds = FiveClassPatchDataset(
        manifest,
        "test",
        augment=False,
    )

    loader = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SegFormerSAR5Class(
        pretrained_name="nvidia/mit-b2",
        num_classes=5,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    preds = []
    targets = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        pred = torch.argmax(logits, dim=1)

        preds.append(pred.cpu())
        targets.append(masks.cpu())

    metrics = compute_metrics(
        torch.cat(preds),
        torch.cat(targets),
    )

    print("\n=== 5-CLASS SEGFORMER TEST RESULTS ===")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:15s} "
            f"IoU={metrics['class_iou'][i]:.4f} "
            f"Dice={metrics['class_dice'][i]:.4f} "
            f"P={metrics['class_precision'][i]:.4f} "
            f"R={metrics['class_recall'][i]:.4f}"
        )

    print(f"\nMean IoU       : {metrics['mean_iou']:.4f}")
    print(f"Mean Dice      : {metrics['mean_dice']:.4f}")
    print(f"Oil IoU        : {metrics['oil_iou']:.4f}")
    print(f"Oil Dice       : {metrics['oil_dice']:.4f}")
    print(f"Oil Precision  : {metrics['oil_precision']:.4f}")
    print(f"Oil Recall     : {metrics['oil_recall']:.4f}")
    print(f"Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")

    out = (
        ROOT / "artifacts" / "reports"
        / "s1_segformer_mit_b2_5class_test_metrics.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
