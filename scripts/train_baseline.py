"""Train and evaluate the Classical Random Forest baseline on SAR pixel features."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from oilspill.baseline.random_forest import RandomForestBaseline
from oilspill.evaluation.confusion import plot_confusion_matrix
from oilspill.utils.io import load_yaml, save_json
from oilspill.utils.logging import get_logger
from oilspill.utils.seed import set_seed

logger = get_logger("TrainBaseline")


def resolve_split_paths(
    split_file: Path,
    raw_dir: Path
) -> tuple:
    """Resolve file paths from scene ID split list."""
    with open(split_file, "r") as f:
        stems = [line.strip() for line in f if line.strip()]

    all_files = list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpg"))
    file_map = {p.stem: p for p in all_files}

    img_paths = []
    msk_paths = []

    for s in stems:
        # Match image and mask
        # Check standard stems
        img_p = file_map.get(s)
        # Check corresponding mask
        msk_p = file_map.get(s) or file_map.get(f"{s}_mask") or file_map.get(f"{s}_label")

        # Also search in labels folder if img_p is in images folder
        if img_p and "images" in str(img_p):
            potential_msk = Path(str(img_p).replace("images", "labels"))
            if potential_msk.exists():
                msk_p = potential_msk

        if img_p and msk_p:
            img_paths.append(img_p)
            msk_paths.append(msk_p)

    return img_paths, msk_paths


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest baseline.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--samples-per-class", type=int, default=30000, help="Pixels sampled per class")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--max-depth", type=int, default=16, help="Maximum tree depth")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    predictions_dir = Path(cfg["paths"]["predictions_dir"]) / "rf"

    for d in [checkpoints_dir, reports_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_split_file = splits_dir / "train_images.txt"
    val_split_file = splits_dir / "val_images.txt"
    test_split_file = splits_dir / "test_images.txt"

    if not train_split_file.exists():
        logger.error(
            f"Split file '{train_split_file}' not found.\n"
            "Please run 'python scripts/prepare_data.py' first."
        )
        return

    train_imgs, train_msks = resolve_split_paths(train_split_file, raw_dir)
    val_imgs, val_msks = resolve_split_paths(val_split_file, raw_dir)
    test_imgs, test_msks = resolve_split_paths(test_split_file, raw_dir)

    logger.info(f"Loaded {len(train_imgs)} train, {len(val_imgs)} validation, {len(test_imgs)} test scenes.")

    # 1. Initialize and Fit Random Forest Baseline
    rf_baseline = RandomForestBaseline(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        random_state=seed,
        window_sizes=(3, 7, 15),
    )

    logger.info("Fitting Random Forest baseline model...")
    rf_baseline.fit(
        image_paths=train_imgs,
        mask_paths=train_msks,
        samples_per_class=args.samples_per_class,
    )

    # Save model checkpoint
    model_checkpoint_path = checkpoints_dir / "random_forest_baseline.joblib"
    rf_baseline.save(model_checkpoint_path)

    # 2. Evaluate on Validation Set
    logger.info("Evaluating on validation set...")
    val_metrics = rf_baseline.evaluate(
        image_paths=val_imgs,
        mask_paths=val_msks,
        output_pred_dir=predictions_dir / "val",
        max_visualize=5,
    )

    # 3. Save Metrics Reports
    metrics_json_path = reports_dir / "baseline_rf_metrics.json"
    metrics_csv_path = reports_dir / "baseline_rf_metrics.csv"
    
    save_json(val_metrics, metrics_json_path)

    # Build clean CSV summary
    records = []
    for cls_name, m in val_metrics["per_class"].items():
        records.append({
            "Class": cls_name,
            "IoU": f"{m['iou']:.4f}",
            "Dice_F1": f"{m['dice']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "GT_Pixels": m["gt_pixels"],
            "Pred_Pixels": m["pred_pixels"],
        })
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(metrics_csv_path, index=False)

    # 4. Save Confusion Matrix Plot
    cm_plot_path = reports_dir / "baseline_rf_confusion_matrix.png"
    plot_confusion_matrix(
        cm=np.array(val_metrics["confusion_matrix"]),
        output_path=cm_plot_path,
        title="Random Forest Baseline Confusion Matrix (Normalized)",
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST BASELINE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall Pixel Accuracy: {val_metrics['overall_pixel_accuracy'] * 100:.2f}%")
    print(f"Mean IoU (mIoU):         {val_metrics['mean_iou']:.4f}")
    print(f"Oil Spill IoU:          {val_metrics['oil_iou']:.4f}  |  Dice/F1: {val_metrics['oil_dice']:.4f}")
    print(f"Look-alike to Oil FA:   {val_metrics['error_analysis']['lookalike_false_alarm_rate'] * 100:.2f}%")
    print("-" * 80)
    print(df_metrics.to_string(index=False))
    print("=" * 80)
    print(f"Checkpoint: {model_checkpoint_path}")
    print(f"JSON:       {metrics_json_path}")
    print(f"CSV:        {metrics_csv_path}")
    print(f"Plot:       {cm_plot_path}")
    print(f"Masks:      {predictions_dir / 'val'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
