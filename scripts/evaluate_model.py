"""Evaluate a saved checkpoint on a split (default: test) and write reports + qualitative examples.

Example:
    python scripts/evaluate_model.py --checkpoint artifacts/checkpoints/e2_unet_cedice_aug_best.pth

Only run this on the TEST split after model selection is frozen (decide using validation metrics).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from oilspill.data.torch_dataset import build_dataloaders
from oilspill.models import build_model
from oilspill.training.config import load_config
from oilspill.training.evaluation import (plot_confusion_matrix, predict_loader, save_example_grid,
                                          save_gallery, select_examples)
from oilspill.training.metrics import detection_metrics, metrics_from_confusion
from oilspill.training.reporting import upsert_experiment
from oilspill.utils.io import save_json
from oilspill.utils.logging import get_logger

logger = get_logger("EvaluateModel")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="Optional; by default the config stored in the checkpoint is used")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-eval-patches", type=int, default=None)
    ap.add_argument("--grid-size", type=int, default=8, help="Rows in the overlay grid")
    ap.add_argument("--gallery-size", type=int, default=24, help="Qualitative example images to save")
    ap.add_argument("--min-oil-pixels", type=int, default=100,
                    help="A scene counts as 'oil detected' if at least this many oil pixels are predicted")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = load_config(args.config) if args.config else ck["config"]
    cfg["model"]["encoder_weights"] = None  # weights come from the checkpoint
    name = ck.get("name") or Path(args.checkpoint).stem.replace("_best", "")
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    nc = int(cfg["model"]["classes"])
    names = ck.get("class_names") or (["background", "oil_spill"] if nc == 2 else list(cfg["dataset"]["classes"].values()))

    data = build_dataloaders(cfg, num_workers=args.num_workers, splits=(args.split,), max_eval_patches=args.max_eval_patches)
    loader = data[args.split]
    ds = loader.dataset
    ds.mean, ds.std = ck.get("norm_mean", ds.mean), ck.get("norm_std", ds.std)  # use TRAIN stats saved with the model
    cm, stats = predict_loader(model, loader, device, nc, amp=bool(cfg["training"].get("amp", True)), desc=args.split)
    m = metrics_from_confusion(cm, names)

    # Scene-level (chip-level) detection: does the model flag oil in scenes that really contain oil?
    st = pd.concat([ds.df[["parent_id"]].reset_index(drop=True), stats], axis=1)
    scene = st.groupby("parent_id").agg(gt=("gt_oil", "sum"), pred=("pred_oil", "sum"))
    det = detection_metrics(scene["gt"] > 0, scene["pred"] >= args.min_oil_pixels)

    rep, pred_dir = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["predictions_dir"])
    tag = f"{name}_{args.split}"
    save_json({"experiment": name, "split": args.split, "pixel_metrics": m, "scene_detection": det,
               "min_oil_pixels": args.min_oil_pixels}, rep / f"{tag}_metrics.json")
    pd.DataFrame([{"class": k, **v} for k, v in m["per_class"].items()]).to_csv(rep / f"{tag}_per_class.csv", index=False)
    plot_confusion_matrix(cm, names, rep / f"{tag}_confusion_matrix.png")
    picks = select_examples(stats, max(args.grid_size, args.gallery_size))
    if picks:
        save_example_grid(model, ds, picks[:args.grid_size], device, pred_dir / f"{tag}_overlay_grid.png")
        n = save_gallery(model, ds, picks[:args.gallery_size], device, pred_dir / f"{tag}_gallery")
    else:
        n = 0
    upsert_experiment(rep / "experiment_matrix.csv", name, {
        f"{args.split}_oil_iou": m.get("oil_iou"), f"{args.split}_oil_dice": m.get("oil_dice"),
        f"{args.split}_oil_precision": m.get("oil_precision"), f"{args.split}_oil_recall": m.get("oil_recall"),
        f"{args.split}_mean_iou": m["mean_iou"], f"{args.split}_pixel_accuracy": m["pixel_accuracy"],
        f"{args.split}_scene_f1": det["f1"], f"{args.split}_scene_recall": det["recall"],
        f"{args.split}_scene_precision": det["precision"]})

    print("\n" + "=" * 70)
    print(f"{name} on {args.split} ({len(ds)} patches, {len(scene)} scenes)")
    print(pd.DataFrame([{"class": k, **{a: (None if b is None else round(b, 4)) for a, b in v.items() if a != 'support_pixels'}}
                        for k, v in m["per_class"].items()]).to_string(index=False))
    print(f"mean IoU {m['mean_iou']:.4f} | oil IoU {m.get('oil_iou', 0):.4f} | oil Dice {m.get('oil_dice', 0):.4f} | "
          f"pixel acc {m['pixel_accuracy']:.4f} (secondary metric)")
    print(f"scene-level detection: precision {det['precision']:.3f} recall {det['recall']:.3f} F1 {det['f1']:.3f} "
          f"(TP {det['tp']} FP {det['fp']} FN {det['fn']} TN {det['tn']})")
    print(f"saved: {rep / (tag + '_metrics.json')}, confusion matrix PNG, overlay grid, {n} gallery images")
    print("=" * 70)


if __name__ == "__main__":
    main()
