"""Train one segmentation experiment.

Example:
    python scripts/train_model.py --config configs/experiments/e2_unet_cedice_aug.yaml
    python scripts/train_model.py --config configs/unet_resnet34.yaml --name smoke --epochs 1 --max-train-patches 64
"""

import argparse
import time
from pathlib import Path

import torch

from oilspill.data.torch_dataset import build_dataloaders
from oilspill.models import build_loss, build_model, count_parameters
from oilspill.training.callbacks import EarlyStopping, ModelCheckpoint
from oilspill.training.config import load_config
from oilspill.training.engine import Trainer
from oilspill.training.reporting import upsert_experiment
from oilspill.utils.io import save_json
from oilspill.utils.logging import get_logger
from oilspill.utils.seed import set_seed

logger = get_logger("TrainModel")


def main():
    ap = argparse.ArgumentParser(description="Train a segmentation model (Chunk 2).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="Experiment name (default: experiment.name from the config)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (use 0 if Windows complains)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--max-train-patches", type=int, default=None, help="Subsample train patches (smoke tests)")
    ap.add_argument("--max-eval-patches", type=int, default=None, help="Subsample val patches (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="Resume from artifacts/checkpoints/<name>_last.pth")
    ap.add_argument("--no-pretrained", action="store_true", help="Random init instead of ImageNet weights")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override config values, e.g. --set model.architecture=deeplabv3plus")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.epochs:
        overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.no_pretrained:
        overrides.append("model.encoder_weights=null")
    cfg = load_config(args.config, overrides)
    name = args.name or cfg.get("experiment", {}).get("name") or f"{cfg['model']['architecture']}_{cfg['model']['encoder_name']}"
    exp_id = cfg.get("experiment", {}).get("id", "-")

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    logger.info(f"Experiment {exp_id} '{name}' on {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CPU is VERY slow, use a GPU)"))

    data = build_dataloaders(cfg, num_workers=args.num_workers, max_train_patches=args.max_train_patches,
                             max_eval_patches=args.max_eval_patches)
    nc = data["num_classes"]
    class_names = ["background", "oil_spill"] if nc == 2 else list(cfg["dataset"]["classes"].values())
    logger.info(f"train patches={len(data['train'].dataset)}  val patches={len(data['val'].dataset)}  "
                f"class_weights={[round(float(w), 3) for w in data['class_weights']]}")

    model = build_model(cfg)
    tcfg = cfg["training"]
    criterion = build_loss(cfg, data["class_weights"])
    opt_name = tcfg.get("optimizer", "AdamW").lower()
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[opt_name]
    optimizer = opt_cls(model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    scheduler = None
    if str(tcfg.get("scheduler", "")).lower() == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(tcfg.get("min_lr", 1e-6)))

    monitor = tcfg.get("monitor", "val_oil_iou")
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    rep_dir = Path(cfg["paths"]["reports_dir"])
    checkpoint = ModelCheckpoint(ckpt_dir, name, monitor=monitor)
    stopper = EarlyStopping(int(tcfg.get("early_stopping_patience", 12)), monitor=monitor)
    trainer = Trainer(model, criterion, optimizer, scheduler, device, nc, class_names, amp=bool(tcfg.get("amp", True)),
                      grad_clip=float(tcfg.get("grad_clip", 1.0)), config=cfg,
                      meta={"name": name, "norm_mean": data["mean"], "norm_std": data["std"],
                            "encoder_weights_used": getattr(model, "encoder_weights_used", None)})
    logger.info(f"model={cfg['model']['architecture']}/{cfg['model']['encoder_name']} params={count_parameters(model) / 1e6:.1f}M "
                f"loss={tcfg['loss']['name']} aug={cfg.get('augmentation', {}).get('enabled', True)} epochs={epochs}")

    res = trainer.fit(data["train"], data["val"], epochs, checkpoint, stopper,
                      history_path=str(rep_dir / f"{name}_history.csv"), resume=args.resume)

    # Reload the best weights and record final validation metrics (test set is NOT touched here).
    best = torch.load(checkpoint.best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    final = trainer.validate(data["val"], desc="final-val")
    save_json({k: v for k, v in final.items()}, rep_dir / f"{name}_val_metrics.json")
    row = {"experiment_id": exp_id, "architecture": cfg["model"]["architecture"], "encoder": cfg["model"]["encoder_name"],
           "encoder_weights": getattr(model, "encoder_weights_used", None), "classes": nc,
           "loss": tcfg["loss"]["name"], "class_weighted": tcfg["loss"].get("use_class_weights", False),
           "augmentation": cfg.get("augmentation", {}).get("enabled", True), "epochs_configured": epochs,
           "epochs_run": res["epochs_run"], "best_epoch": res["best_epoch"], "params_millions": round(count_parameters(model) / 1e6, 2),
           "train_time_min": round(res["train_time_min"], 1), "seed": seed, "device": str(device),
           "val_oil_iou": final.get("val_oil_iou"), "val_oil_dice": final.get("val_oil_dice"),
           "val_oil_precision": final.get("val_oil_precision"), "val_oil_recall": final.get("val_oil_recall"),
           "val_mean_iou": final.get("val_mean_iou"), "val_pixel_accuracy": final.get("val_pixel_accuracy"),
           "checkpoint": str(checkpoint.best_path), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    upsert_experiment(rep_dir / "experiment_matrix.csv", name, row)
    print("\n" + "=" * 70)
    print(f"DONE {exp_id} {name}: best epoch {res['best_epoch']} | val oil IoU {final.get('val_oil_iou'):.4f} | "
          f"val oil Dice {final.get('val_oil_dice'):.4f} | {res['train_time_min']:.1f} min")
    print(f"checkpoint: {checkpoint.best_path}\nhistory:    {rep_dir / (name + '_history.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
