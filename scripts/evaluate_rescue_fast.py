
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_rescue_pipeline import make_model, ProjectPatchDataset

@torch.inference_mode()
def collect_predictions(model, loader, device, tta=True):
    model.eval()
    all_probs = []
    all_targets = []

    for x, y in tqdm(loader, desc="Inference", leave=False):
        x = x.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]

        if tta:
            xh = torch.flip(x, dims=[3])  # horizontal flip (W)
            ph = torch.softmax(model(xh), dim=1)[:, 1]
            ph = torch.flip(ph, dims=[2])  # undo W flip on BHW

            xv = torch.flip(x, dims=[2])  # vertical flip (H)
            pv = torch.softmax(model(xv), dim=1)[:, 1]
            pv = torch.flip(pv, dims=[1])  # undo H flip on BHW

            probs = (probs + ph + pv) / 3.0

        all_probs.append(probs.float().cpu().numpy())
        all_targets.append((y == 1).cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)

def binary_metrics(probs, targets, threshold):
    pred = probs >= threshold
    gt = targets.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = make_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    val_ds = ProjectPatchDataset(args.project, "val", False)
    test_ds = ProjectPatchDataset(args.project, "test", False)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    use_tta = not args.no_tta
    print(f"Validation patches: {len(val_ds)}")
    print(f"Test patches: {len(test_ds)}")
    print(f"TTA: {use_tta}")
    print("Running validation inference ONCE and caching probabilities...")

    val_probs, val_targets = collect_predictions(model, val_loader, device, use_tta)
    default_val = binary_metrics(val_probs, val_targets, 0.5)
    print("Validation @ threshold 0.50:", default_val)

    thresholds = np.arange(0.20, 0.801, 0.025)
    best_t = 0.5
    best_iou = -1.0
    best_metrics = None

    # Threshold sweep uses cached validation probabilities — no repeated model inference.
    for t in thresholds:
        m = binary_metrics(val_probs, val_targets, float(t))
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_t = float(t)
            best_metrics = m

    print(f"BEST VALIDATION THRESHOLD: {best_t:.3f}")
    print("BEST VALIDATION METRICS:", best_metrics)

    print("Running test inference ONCE with the frozen model...")
    test_probs, test_targets = collect_predictions(model, test_loader, device, use_tta)
    final_test = binary_metrics(test_probs, test_targets, best_t)

    print("FINAL TEST (threshold frozen from validation):")
    print(final_test)

    out = Path(args.checkpoint).resolve().parent
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tta": use_tta,
        "best_validation_threshold": best_t,
        "validation_default_0.5": default_val,
        "validation_best": best_metrics,
        "test_final": final_test,
    }
    out_file = out / "project_transfer_eval_fast.json"
    out_file.write_text(json.dumps(result, indent=2))
    print("Saved:", out_file)

if __name__ == "__main__":
    main()
