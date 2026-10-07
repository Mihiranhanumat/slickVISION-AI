"""Run the whole E0-E4 matrix sequentially (Windows/Linux friendly), then evaluate on the test set.

Examples:
    python scripts/run_experiments.py                         # everything, configs' epochs (30)
    python scripts/run_experiments.py --epochs 15             # faster
    python scripts/run_experiments.py --only E2 E3            # subset
    python scripts/run_experiments.py --skip-eval             # train only (evaluate later)
Finished experiments (best checkpoint exists) are skipped, so you can safely re-run after an interruption.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONFIGS = {
    "E0": "configs/experiments/e0_unet_ce_noaug.yaml",
    "E1": "configs/experiments/e1_unet_cedice_noaug.yaml",
    "E2": "configs/experiments/e2_unet_cedice_aug.yaml",
    "E3": "configs/experiments/e3_deeplab_cedice_aug.yaml",
    "E4": "configs/experiments/e4_best_focaltversky_aug.yaml",
}
NAMES = {"E0": "e0_unet_ce_noaug", "E1": "e1_unet_cedice_noaug", "E2": "e2_unet_cedice_aug",
         "E3": "e3_deeplab_cedice_aug", "E4": "e4_best_focaltversky_aug"}
CK = Path("artifacts/checkpoints")
MATRIX = Path("artifacts/reports/experiment_matrix.csv")


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-train-patches", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Re-train even if a checkpoint exists")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    extra = ["--num-workers", str(args.num_workers)]
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.max_train_patches:
        extra += ["--max-train-patches", str(args.max_train_patches)]

    for exp in args.only:
        done = (CK / f"{NAMES[exp]}_best.pth").exists() and not args.force
        if done:
            print(f"[skip] {exp} already trained")
            continue
        cmd = [sys.executable, "scripts/train_model.py", "--config", CONFIGS[exp], *extra]
        if exp == "E4" and MATRIX.exists():  # E4 uses the better architecture of E2 / E3 (by VALIDATION oil IoU)
            m = pd.read_csv(MATRIX).set_index("experiment_id")
            if {"E2", "E3"} <= set(m.index) and m.loc["E3", "val_oil_iou"] > m.loc["E2", "val_oil_iou"]:
                cmd += ["--set", "model.architecture=deeplabv3plus"]
                print("[E4] DeepLabV3+ won on validation -> E4 uses DeepLabV3+")
            else:
                print("[E4] U-Net won (or E3 missing) -> E4 uses U-Net")
        run(cmd)

    if not args.skip_eval:
        for exp in args.only:
            ck = CK / f"{NAMES[exp]}_best.pth"
            if ck.exists():
                run([sys.executable, "scripts/evaluate_model.py", "--checkpoint", str(ck), "--num-workers", str(args.num_workers)])
        run([sys.executable, "scripts/compare_experiments.py"])


if __name__ == "__main__":
    main()
