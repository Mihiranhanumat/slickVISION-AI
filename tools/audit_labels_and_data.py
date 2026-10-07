import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

def load_mask(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        from PIL import Image
        return np.array(Image.open(path))
    raise ValueError(f"Unsupported mask format: {path}")

def resolve_path(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return root / p

def main():
    ap = argparse.ArgumentParser(
        description="Audit slickVISION-AI masks and manifest WITHOUT changing any data."
    )
    ap.add_argument("--root", default=".", help="Repository root (default: current directory)")
    ap.add_argument("--sample-per-split", type=int, default=500,
                    help="Masks to inspect per split; use 0 for all.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest = root / "data" / "splits" / "patch_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"ERROR: manifest not found: {manifest}")

    df = pd.read_csv(manifest)
    print("=" * 72)
    print("slickVISION-AI LABEL / DATA AUDIT — NO FILES WILL BE MODIFIED")
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Manifest: {manifest}")
    print(f"Rows: {len(df):,}")
    print()

    required = {"split", "mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: missing required columns: {sorted(missing)}")

    print("SPLIT COUNTS")
    print(df["split"].value_counts().to_string())
    print()

    for col in ["oil_pixels", "lookalike_pixels", "ship_pixels", "land_pixels", "sea_pixels"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            print(f"{col:18s} total={int(vals.sum()):,}  positive-patches={(vals>0).sum():,}")
    print()

    print("CHECKING ACTUAL MASK VALUES")
    all_values = set()
    per_split = {}

    for split in ["train", "val", "test"]:
        sub = df[df["split"].astype(str) == split]
        if args.sample_per_split == 0:
            sample = sub
        else:
            sample = sub.head(min(args.sample_per_split, len(sub)))

        split_values = set()
        checked = 0
        missing_paths = 0
        bad_shapes = 0

        for raw in sample["mask_path"].tolist():
            p = resolve_path(root, raw)
            if not p.exists():
                missing_paths += 1
                continue
            try:
                arr = load_mask(p)
                split_values.update(np.unique(arr).astype(int).tolist())
                checked += 1
                if arr.ndim != 2:
                    bad_shapes += 1
            except Exception as e:
                print(f"  WARN: failed {p}: {e}")

        per_split[split] = sorted(split_values)
        all_values.update(split_values)
        print(
            f"{split:5s} checked={checked:4d}  missing={missing_paths:4d}  "
            f"bad_shapes={bad_shapes:3d}  unique_ids={sorted(split_values)}"
        )

    print()
    print("GLOBAL OBSERVED MASK IDS:", sorted(all_values))

    print()
    if all(v in {0, 1} for v in all_values):
        print("CONCLUSION: DATA APPEARS BINARY (0/1) in the inspected masks.")
        print("Next training route: binary oil-spill optimization + external hard-negative pretraining/fine-tuning.")
    elif all(v in {0, 1, 2, 3, 4} for v in all_values) and any(v >= 2 for v in all_values):
        print("CONCLUSION: DATA CONTAINS THE INTENDED 5-CLASS LABEL SET (0–4).")
        print("Next training route: corrected 5-class training + class-aware sampling + stronger architecture.")
    else:
        print("CONCLUSION: LABEL SET NEEDS MANUAL REVIEW.")
        print("Observed IDs are not a clean binary {0,1} or full 5-class {0..4} set.")

    print()
    print("IMPORTANT: This script does NOT run prepare_data.py, does NOT change masks,")
    print("does NOT alter train/val/test splits, and does NOT touch the official test set.")
    print("=" * 72)

if __name__ == "__main__":
    main()
