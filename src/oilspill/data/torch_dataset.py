"""PyTorch Dataset + DataLoader builders for the 256x256 patches produced by scripts/prepare_data.py.

Chunk 1 stores patches as PNG files and describes them in data/splits/patch_manifest.csv
(columns: patch_id, parent_id, split, image_path, mask_path, has_oil, oil_pixels, total_pixels, ...).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import get_training_transforms

_COUNT_COLS = ["sea_pixels", "oil_pixels", "lookalike_pixels", "ship_pixels", "land_pixels"]


def load_norm_stats(splits_dir: Union[str, Path]) -> Tuple[float, float]:
    """Training-set mean/std written by prepare_data.py (falls back to 0.5 / 0.25)."""
    f = Path(splits_dir) / "norm_stats.json"
    if f.exists():
        import json
        d = json.loads(f.read_text())
        return float(d.get("sar_mean", 0.5)), float(max(d.get("sar_std", 0.25), 1e-3))
    return 0.5, 0.25


class OilSpillPatchDataset(Dataset):
    """Reads (image, mask) PNG patches for one split of the patch manifest.

    Returns image (1, H, W) float32 standardised with TRAIN statistics, and mask (H, W) int64.
    num_classes == 2 collapses the mask to {0: background, 1: oil}; otherwise IDs 0-4 are kept.
    """

    def __init__(self, manifest: Union[pd.DataFrame, str, Path], split: str, num_classes: int = 2,
                 augment: bool = False, aug_cfg: Optional[Dict[str, Any]] = None,
                 mean: float = 0.5, std: float = 0.25, root: Union[str, Path, None] = None,
                 max_patches: Optional[int] = None, seed: int = 42):
        df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
        df = df[df["split"] == split].reset_index(drop=True)
        if max_patches and max_patches < len(df):
            df = df.sample(n=max_patches, random_state=seed).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No patches found for split '{split}'. Did you run scripts/prepare_data.py?")
        self.df, self.split, self.num_classes = df, split, num_classes
        self.mean, self.std = mean, std
        self.root = Path(root) if root else Path.cwd()
        sp = (aug_cfg or {}).get("spatial", {})
        it = (aug_cfg or {}).get("intensity", {})
        self.transform = None
        if augment:
            self.transform = get_training_transforms(
                horizontal_flip=sp.get("horizontal_flip_prob", 0.5),
                vertical_flip=sp.get("vertical_flip_prob", 0.5),
                random_rotate90=sp.get("random_rotate90_prob", 0.5),
                brightness_contrast=it.get("brightness_contrast_prob", 0.3),
            )

    def __len__(self) -> int:
        return len(self.df)

    def _path(self, p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else self.root / pp)

    def read_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Un-normalised uint8 image and processed mask (used for visualisation)."""
        row = self.df.iloc[idx]
        img = cv2.imread(self._path(row["image_path"]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(self._path(row["mask_path"]), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            raise FileNotFoundError(f"Missing patch files for {row['patch_id']}")
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = (msk == 1).astype(np.uint8) if self.num_classes == 2 else np.clip(msk, 0, self.num_classes - 1)
        return img, msk

    def __getitem__(self, idx: int):
        img, msk = self.read_raw(idx)
        if self.transform is not None:
            out = self.transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"]
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(np.ascontiguousarray(msk)).long()


def get_class_weights(manifest: pd.DataFrame, num_classes: int = 2, split: str = "train",
                      clip: Tuple[float, float] = (0.1, 50.0)) -> torch.Tensor:
    """Median-frequency class weights from TRAIN patches only (no val/test leakage)."""
    df = manifest[manifest["split"] == split]
    if num_classes == 2:
        oil = float(df["oil_pixels"].sum())
        counts = np.array([float(df["total_pixels"].sum()) - oil, oil])
    else:
        counts = np.array([float(df[c].sum()) for c in _COUNT_COLS][:num_classes])
    freq = counts / max(counts.sum(), 1.0)
    present = freq > 0
    w = np.ones(num_classes)
    if present.any():
        w[present] = np.clip(np.median(freq[present]) / freq[present], *clip)
    return torch.tensor(w, dtype=torch.float32)


def build_dataloaders(cfg: Dict[str, Any], num_workers: int = 2, max_train_patches: Optional[int] = None,
                      max_eval_patches: Optional[int] = None, splits=("train", "val"),
                      root: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Create DataLoaders for the requested splits plus class weights and normalisation stats."""
    splits_dir = Path(cfg["paths"]["splits_dir"])
    manifest_path = (Path(root) / splits_dir if root else splits_dir) / "patch_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run: python scripts/prepare_data.py")
    manifest = pd.read_csv(manifest_path)
    nc = int(cfg["model"]["classes"])
    mean, std = load_norm_stats(manifest_path.parent)
    tcfg = cfg["training"]
    bs = int(tcfg["batch_size"])
    aug_cfg = cfg.get("augmentation", {})
    use_aug = bool(aug_cfg.get("enabled", True))
    pin = torch.cuda.is_available()
    out: Dict[str, Any] = {"mean": mean, "std": std, "num_classes": nc,
                           "class_weights": get_class_weights(manifest, nc), "manifest": manifest}
    for sp in splits:
        train = sp == "train"
        ds = OilSpillPatchDataset(manifest, sp, nc, augment=train and use_aug, aug_cfg=aug_cfg,
                                  mean=mean, std=std, root=root,
                                  max_patches=max_train_patches if train else max_eval_patches,
                                  seed=cfg.get("project", {}).get("seed", 42))
        out[sp] = DataLoader(ds, batch_size=bs if train else int(tcfg.get("eval_batch_size", bs)),
                             shuffle=train, drop_last=train and len(ds) > bs, num_workers=num_workers,
                             pin_memory=pin, persistent_workers=num_workers > 0)
    return out
