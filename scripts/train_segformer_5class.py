from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilspill.models.segformer_sar_5class import SegFormerSAR5Class


CLASS_NAMES = [
    "sea_surface",
    "oil_spill",
    "look_alike",
    "ship",
    "land",
]
NUM_CLASSES = 5


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        ROOT / p,
        ROOT / "data" / p,
        ROOT / "data" / "patches" / p,
        ROOT / "data" / "processed" / p,
        ROOT / "data" / "splits" / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Could not resolve manifest path: {value}")


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower().strip(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path))


class FiveClassPatchDataset(Dataset):
    """
    Reads the EXISTING patch manifest and preserves class IDs 0..4 exactly.

    No binary conversion is performed.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
    ):
        df = pd.read_csv(manifest_path)

        split_col = choose_column(
            df, ["split", "set", "partition", "subset", "split_name"]
        )
        image_col = choose_column(
            df,
            [
                "image_path",
                "image",
                "patch_path",
                "patch",
                "image_file",
                "image_filepath",
                "sar_path",
                "input_path",
            ],
        )
        mask_col = choose_column(
            df,
            [
                "mask_path",
                "mask",
                "mask_file",
                "mask_filepath",
                "label_path",
                "label",
                "target_path",
                "target",
            ],
        )

        if split_col is None or image_col is None or mask_col is None:
            raise ValueError(
                "Manifest columns could not be identified. "
                f"Columns: {list(df.columns)}"
            )

        self.df = df[
            df[split_col].astype(str).str.lower().str.strip() == split.lower()
        ].reset_index(drop=True)

        self.image_col = image_col
        self.mask_col = mask_col
        self.image_size = image_size
        self.augment = augment

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = np.squeeze(
            load_array(resolve_path(row[self.image_col]))
        ).astype(np.float32)

        mask = np.squeeze(
            load_array(resolve_path(row[self.mask_col]))
        )

        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays. Got image={image.shape}, mask={mask.shape}"
            )

        # Preserve original five class IDs.
        unique_mask = np.unique(mask)
        if np.any(unique_mask < 0) or np.any(unique_mask > 4):
            raise ValueError(
                f"Mask contains invalid class IDs: {unique_mask}"
            )

        image_t = torch.from_numpy(image).unsqueeze(0).float()

        # IMPORTANT: no mask > 0 conversion here.
        mask_t = torch.from_numpy(mask.astype(np.int64)).long()

        if float(image_t.max()) > 1.0:
            image_t = image_t / 255.0

        if tuple(image_t.shape[-2:]) != (self.image_size, self.image_size):
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_t = F.interpolate(
                mask_t[None, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            )[0, 0].long()

        if self.augment:
            if random.random() < 0.5:
                image_t = torch.flip(image_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[1])

            if random.random() < 0.5:
                image_t = torch.flip(image_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[0])

            k = random.randrange(4)
            if k:
                image_t = torch.rot90(image_t, k, dims=[1, 2])
                mask_t = torch.rot90(mask_t, k, dims=[0, 1])

            if random.random() < 0.30:
                scale = random.uniform(0.90, 1.10)
                shift = random.uniform(-0.05, 0.05)
                image_t = torch.clamp(image_t * scale + shift, 0.0, 1.0)

        return image_t, mask_t


class CombinedCEDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
        )

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(
            target,
            num_classes=NUM_CLASSES,
        ).permute(0, 3, 1, 2).float()

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denominator = (
            probs.sum(dim=(0, 2, 3))
            + one_hot.sum(dim=(0, 2, 3))
        )

        per_class_dice = (
            (2.0 * intersection + 1e-6)
            / (denominator + 1e-6)
        )

        dice = per_class_dice.mean()

        return (
            self.ce_weight * ce
            + self.dice_weight * (1.0 - dice)
        )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.reshape(-1).long().cpu()
    target = target.reshape(-1).long().cpu()

    ids = target * NUM_CLASSES + pred
    cm = torch.bincount(
        ids,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES).float()

    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return {
        "class_iou": iou.numpy().tolist(),
        "class_dice": dice.numpy().tolist(),
        "class_precision": precision.numpy().tolist(),
        "class_recall": recall.numpy().tolist(),
        "mean_iou": float(iou.mean()),
        "mean_dice": float(dice.mean()),
        "oil_iou": float(iou[1]),
        "oil_dice": float(dice[1]),
        "oil_precision": float(precision[1]),
        "oil_recall": float(recall[1]),
        "pixel_accuracy": float(tp.sum() / (cm.sum() + 1e-7)),
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, targets = [], []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(masks.cpu())

    return compute_metrics(torch.cat(preds), torch.cat(targets))


def load_class_weights() -> torch.Tensor:
    path = ROOT / "data" / "splits" / "class_weights.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected training class weights at {path}"
        )

    data = json.loads(path.read_text())

    # class_weights.json uses semantic names.
    weights = torch.tensor(
        [
            float(data.get("sea_surface", 1.0)),
            float(data.get("oil_spill", 1.0)),
            float(data.get("look_alike", 1.0)),
            float(data.get("ship", 1.0)),
            float(data.get("land", 1.0)),
        ],
        dtype=torch.float32,
    )

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/splits/patch_manifest.csv",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    manifest = resolve_path(args.manifest)

    train_ds = FiveClassPatchDataset(
        manifest,
        "train",
        augment=True,
    )
    val_ds = FiveClassPatchDataset(
        manifest,
        "val",
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Train patches: {len(train_ds)}")
    print(f"Validation patches: {len(val_ds)}")
    print("Test patches: NOT USED")
    print("Classes: 5")
    print("0 sea_surface | 1 oil_spill | 2 look_alike | 3 ship | 4 land")

    class_weights = load_class_weights()
    print("Class weights:", class_weights.tolist())

    model = SegFormerSAR5Class(
        pretrained_name="nvidia/mit-b2",
        num_classes=5,
    ).to(device)

    criterion = CombinedCEDiceLoss(
        class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    report_dir = ROOT / "artifacts" / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    bad_epochs = 0
    patience = 5
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        bar = tqdm(
            train_loader,
            desc=f"SegFormer-5 epoch {epoch}/{args.epochs}",
        )

        for images, masks in bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val = validate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            "mean_iou": val["mean_iou"],
            "mean_dice": val["mean_dice"],
            "oil_iou": val["oil_iou"],
            "oil_dice": val["oil_dice"],
            "oil_precision": val["oil_precision"],
            "oil_recall": val["oil_recall"],
            "pixel_accuracy": val["pixel_accuracy"],
        }

        for i, name in enumerate(CLASS_NAMES):
            row[f"{name}_iou"] = val["class_iou"][i]
            row[f"{name}_dice"] = val["class_dice"][i]

        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"mean_mIoU={val['mean_iou']:.4f} | "
            f"oil_IoU={val['oil_iou']:.4f} | "
            f"oil_Dice={val['oil_dice']:.4f} | "
            f"pixel_acc={val['pixel_accuracy']:.4f}"
        )

        # Model selection follows project methodology: validation mean IoU.
        if val["mean_iou"] > best_iou:
            best_iou = val["mean_iou"]
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "segformer_mit_b2_5class",
                    "pretrained_name": "nvidia/mit-b2",
                    "classes": 5,
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_metrics": val,
                    "seed": args.seed,
                },
                checkpoint_dir / "s1_segformer_mit_b2_5class_best.pth",
            )

            print(
                f"  NEW BEST -> validation mean IoU {best_iou:.4f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    history_path = (
        report_dir / "s1_segformer_mit_b2_5class_history.csv"
    )
    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("\n=== 5-CLASS SCREENING COMPLETE ===")
    print(f"Best validation mean IoU: {best_iou:.4f}")
    print(f"History: {history_path}")
    print(
        "Checkpoint:",
        checkpoint_dir / "s1_segformer_mit_b2_5class_best.pth",
    )


if __name__ == "__main__":
    main()
