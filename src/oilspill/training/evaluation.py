"""Evaluation helpers: loader inference, confusion matrix plot, qualitative example galleries."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.dataset import mask_to_rgb
from .metrics import confusion_from_tensors


@torch.no_grad()
def predict_loader(model, loader, device, num_classes: int, amp: bool = False, desc: str = "eval"):
    """Run the model over a NON-shuffled loader.

    Returns (confusion_matrix [C, C] ndarray, per-patch DataFrame with gt_oil / pred_oil / tp_oil in loader order).
    """
    device = torch.device(device)
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    rows = []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        pred = logits.argmax(1)
        cm += confusion_from_tensors(pred, y, num_classes).cpu()
        gt, pr = (y == 1).flatten(1), (pred == 1).flatten(1)
        for a, b, c in zip(gt.sum(1).tolist(), pr.sum(1).tolist(), (gt & pr).sum(1).tolist()):
            rows.append({"gt_oil": a, "pred_oil": b, "tp_oil": c})
    return cm.numpy(), pd.DataFrame(rows)


def plot_confusion_matrix(cm, class_names: List[str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j] * 100:.1f}%\n({int(cm[i, j]):,})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth (row-normalised)")
    ax.set_title("Pixel confusion matrix")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_examples(stats: pd.DataFrame, k: int = 8) -> List[Tuple[int, str]]:
    """Pick qualitative examples: some good detections, worst misses (FN) and worst false alarms (FP)."""
    s = stats.copy()
    s["union"] = s["gt_oil"] + s["pred_oil"] - s["tp_oil"]
    s["iou"] = np.where(s["union"] > 0, s["tp_oil"] / s["union"].clip(lower=1), 1.0)
    s["recall"] = np.where(s["gt_oil"] > 0, s["tp_oil"] / s["gt_oil"].clip(lower=1), 1.0)
    n_good, n_fn = max(k // 4, 1), max(k * 3 // 8, 1)
    n_fp = max(k - n_good - n_fn, 1)
    good = s[(s.gt_oil > 0)].sort_values("iou", ascending=False).head(n_good).index
    fn = s[(s.gt_oil > 0)].sort_values("recall").head(n_fn).index
    fp = s[(s.gt_oil == 0)].sort_values("pred_oil", ascending=False).head(n_fp).index
    out = [(int(i), "good") for i in good] + [(int(i), "missed oil (FN)") for i in fn]
    out += [(int(i), "false alarm (FP)") for i in fp if s.loc[i, "pred_oil"] > 0]
    return out


@torch.no_grad()
def compose_row(model, dataset, idx: int, device, label: str = "") -> np.ndarray:
    """One RGB row: SAR | ground truth | prediction | error map (TP cyan, FP red, FN yellow)."""
    img, gt = dataset.read_raw(idx)
    x, _ = dataset[idx]
    pred = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    err = np.zeros_like(rgb)
    err[(pred == 1) & (gt == 1)] = (0, 255, 255)
    err[(pred == 1) & (gt != 1)] = (255, 0, 0)
    err[(pred != 1) & (gt == 1)] = (255, 255, 0)
    row = np.hstack([rgb, mask_to_rgb(gt), mask_to_rgb(pred), err])
    cv2.putText(row, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return row


def save_example_grid(model, dataset, picks: List[Tuple[int, str]], device, path) -> None:
    model.eval()
    rows = [compose_row(model, dataset, i, device, f"{kind} | {dataset.df.loc[i, 'patch_id']}") for i, kind in picks]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def save_gallery(model, dataset, picks: List[Tuple[int, str]], device, out_dir) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for n, (i, kind) in enumerate(picks):
        row = compose_row(model, dataset, i, device, f"{kind} | {dataset.df.loc[i, 'patch_id']}")
        cv2.imwrite(str(out_dir / f"{n:03d}_{kind.split()[0]}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    return len(picks)
