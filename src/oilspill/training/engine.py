"""Training engine: AMP, gradient clipping, cosine LR, checkpointing, early stopping, resume."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint, get_lr
from .metrics import confusion_from_tensors, metrics_from_confusion


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Owns the model / loss / optimiser and runs train_one_epoch, validate and fit."""

    def __init__(self, model, criterion, optimizer, scheduler=None, device="cpu", num_classes: int = 2,
                 class_names: Optional[List[str]] = None, amp: bool = True, grad_clip: Optional[float] = 1.0,
                 show_progress: bool = True, config: Optional[dict] = None, meta: Optional[dict] = None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer, self.scheduler = optimizer, scheduler
        self.num_classes, self.class_names = num_classes, class_names
        self.use_amp = bool(amp) and self.device.type == "cuda"
        self.scaler = _make_scaler(self.use_amp)
        self.grad_clip, self.show_progress = grad_clip, show_progress
        self.config, self.meta = config or {}, meta or {}

    # ------------------------------------------------------------------ one epoch
    def train_one_epoch(self, loader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        total, n, skipped = 0.0, 0, 0
        bar = tqdm(loader, desc=f"train {epoch:03d}", leave=False, disable=not self.show_progress)
        for x, y in bar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
            loss = self.criterion(logits.float(), y)
            if not torch.isfinite(loss):  # guard against rare AMP overflows
                skipped += 1
                continue
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
            bar.set_postfix(loss=f"{total / max(n, 1):.4f}")
        return {"train_loss": total / max(n, 1), "skipped_batches": skipped}

    @torch.no_grad()
    def validate(self, loader, desc: str = "val") -> Dict[str, Any]:
        self.model.eval()
        total, n = 0.0, 0
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        for x, y in tqdm(loader, desc=desc, leave=False, disable=not self.show_progress):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
            total += self.criterion(logits.float(), y).item() * x.size(0)
            n += x.size(0)
            cm += confusion_from_tensors(logits.argmax(1), y, self.num_classes).cpu()
        m = metrics_from_confusion(cm.numpy(), self.class_names)
        out = {"val_loss": total / max(n, 1), "confusion_matrix": m["confusion_matrix"]}
        for k in ("mean_iou", "mean_dice", "pixel_accuracy", "oil_iou", "oil_dice", "oil_precision", "oil_recall"):
            if k in m:
                out[f"val_{k}"] = m[k]
        return out

    # ------------------------------------------------------------------ checkpoints
    def _light(self, epoch: int, metrics: dict) -> dict:
        return {"model_state": self.model.state_dict(), "config": self.config, "epoch": epoch,
                "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "class_names": self.class_names, **self.meta}

    def _full(self, epoch: int, history: list, ckpt: ModelCheckpoint, es: Optional[EarlyStopping]) -> dict:
        return {"model_state": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "scaler": self.scaler.state_dict(), "epoch": epoch, "history": history,
                "best": ckpt.best, "best_epoch": ckpt.best_epoch,
                "es": (es.best, es.counter) if es else None, "config": self.config,
                "class_names": self.class_names, **self.meta}

    def _resume(self, path, ckpt: ModelCheckpoint, es: Optional[EarlyStopping]):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        ckpt.best, ckpt.best_epoch = ck.get("best"), ck.get("best_epoch", 0)
        if es and ck.get("es"):
            es.best, es.counter = ck["es"]
        print(f"[resume] continuing from epoch {ck['epoch'] + 1}")
        return ck["epoch"] + 1, ck["history"]

    # ------------------------------------------------------------------ main loop
    def fit(self, train_loader, val_loader, epochs: int, checkpoint: ModelCheckpoint,
            early_stopping: Optional[EarlyStopping] = None, history_path: Optional[str] = None,
            resume: bool = False) -> Dict[str, Any]:
        start, history = 1, []
        if resume and checkpoint.last_path.exists():
            start, history = self._resume(checkpoint.last_path, checkpoint, early_stopping)
        t_start, stopped_early = time.time(), False
        for epoch in range(start, epochs + 1):
            t0 = time.time()
            lr = get_lr(self.optimizer)
            tr = self.train_one_epoch(train_loader, epoch)
            va = self.validate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            row = {"epoch": epoch, "lr": lr, "train_loss": tr["train_loss"],
                   **{k: v for k, v in va.items() if k != "confusion_matrix"},
                   "epoch_time_s": time.time() - t0}
            history.append(row)
            improved = checkpoint.step(epoch, row, lambda: self._light(epoch, row))
            checkpoint.save_last(self._full(epoch, history, checkpoint, early_stopping))
            if history_path:
                Path(history_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(history_path, index=False)
            print(f"epoch {epoch:03d}/{epochs} | train_loss {row['train_loss']:.4f} | val_loss {row['val_loss']:.4f} "
                  f"| val_oil_iou {row.get('val_oil_iou', float('nan')):.4f} | val_mIoU {row['val_mean_iou']:.4f} "
                  f"| lr {lr:.2e} | {row['epoch_time_s']:.0f}s{'  *best*' if improved else ''}", flush=True)
            if early_stopping is not None and early_stopping.step(row):
                print(f"[early stopping] no improvement in {early_stopping.monitor} for "
                      f"{early_stopping.patience} epochs.")
                stopped_early = True
                break
        return {"best_value": checkpoint.best, "best_epoch": checkpoint.best_epoch, "epochs_run": len(history),
                "stopped_early": stopped_early, "train_time_min": (time.time() - t_start) / 60.0,
                "history": pd.DataFrame(history)}
