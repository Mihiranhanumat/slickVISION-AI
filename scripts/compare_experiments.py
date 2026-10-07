"""Build the final comparison table: Random Forest baseline vs E0-E4.

Reads  artifacts/reports/experiment_matrix.csv  (+ baseline_rf_metrics.json if present)
Writes artifacts/reports/experiment_comparison.md and experiment_comparison.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from oilspill.utils.io import load_json


def fmt(v):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{float(v):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="artifacts/reports")
    args = ap.parse_args()
    rep = Path(args.reports_dir)
    mat_path = rep / "experiment_matrix.csv"
    if not mat_path.exists():
        raise SystemExit(f"{mat_path} not found - train at least one experiment first.")
    df = pd.read_csv(mat_path).sort_values("experiment_id")
    cols = ["val_oil_iou", "val_oil_dice", "test_oil_iou", "test_oil_dice", "test_oil_precision",
            "test_oil_recall", "test_mean_iou", "test_scene_f1"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    rows = []
    rf_json = rep / "baseline_rf_metrics.json"
    if rf_json.exists():
        rf = load_json(rf_json)
        rows.append({"ID": "RF", "Model": "Random Forest (Chunk 1)", "Loss": "balanced sampling", "Aug": "-",
                     "val_oil_iou": rf.get("oil_iou"), "val_oil_dice": rf.get("oil_dice"),
                     "test_oil_iou": None, "test_oil_dice": None, "test_oil_precision": None,
                     "test_oil_recall": None, "test_mean_iou": None, "test_scene_f1": None})
    for _, r in df.iterrows():
        rows.append({"ID": r["experiment_id"], "Model": f"{r['architecture']}/{r['encoder']}", "Loss": r["loss"],
                     "Aug": "yes" if str(r["augmentation"]) == "True" else "no", **{c: r[c] for c in cols}})
    out = pd.DataFrame(rows)
    out.to_csv(rep / "experiment_comparison.csv", index=False)

    best = df.loc[df["val_oil_iou"].astype(float).idxmax()] if df["val_oil_iou"].notna().any() else None
    lines = ["# Experiment comparison", "",
             "Model selection uses VALIDATION oil IoU. Test columns are filled by scripts/evaluate_model.py and "
             "must be computed once, after the choice is frozen.", "",
             "| ID | Model | Loss | Aug | Val oil IoU | Val oil Dice | Test oil IoU | Test oil Dice | Test precision | Test recall | Test mIoU | Scene F1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r['ID']} | {r['Model']} | {r['Loss']} | {r['Aug']} | " + " | ".join(fmt(r[c]) for c in cols) + " |")
    if best is not None:
        lines += ["", f"**Best on validation oil IoU:** {best['experiment_id']} ({best['name']}) = {float(best['val_oil_iou']):.4f}"]
    lines += ["", "Caveat: ground-truth masks are Otsu-threshold pseudo-labels derived from chip-level labels (Chunk 1), "
              "so IoU measures agreement with those masks. The RF row is a pixel-level validation score from Chunk 1 "
              "and is only indicative (different training pipeline, 5-class setup)."]
    (rep / "experiment_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
