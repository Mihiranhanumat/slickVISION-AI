"""Training callbacks: early stopping, checkpointing, learning-rate helpers."""

from pathlib import Path
from typing import Callable, Dict, Optional

import torch


def _better(value: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    return value > best + min_delta if mode == "max" else value < best - min_delta


class EarlyStopping:
    """Stop when `monitor` has not improved for `patience` epochs."""

    def __init__(self, patience: int = 12, monitor: str = "val_oil_iou", mode: str = "max",
                 min_delta: float = 1e-4):
        self.patience, self.monitor, self.mode, self.min_delta = patience, monitor, mode, min_delta
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, metrics: Dict[str, float]) -> bool:
        """Returns True when training should stop."""
        v = metrics[self.monitor]
        if _better(v, self.best, self.mode, self.min_delta):
            self.best, self.counter = v, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    """Keeps the best weights (by `monitor`) as {name}_best.pth and the latest full state as {name}_last.pth."""

    def __init__(self, save_dir, name: str, monitor: str = "val_oil_iou", mode: str = "max"):
        self.dir = Path(save_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.dir / f"{name}_best.pth"
        self.last_path = self.dir / f"{name}_last.pth"
        self.monitor, self.mode = monitor, mode
        self.best: Optional[float] = None
        self.best_epoch = 0

    def step(self, epoch: int, metrics: Dict[str, float], payload_fn: Callable[[], dict]) -> bool:
        v = metrics[self.monitor]
        if _better(v, self.best, self.mode):
            self.best, self.best_epoch = v, epoch
            torch.save(payload_fn(), self.best_path)
            return True
        return False

    def save_last(self, payload: dict) -> None:
        torch.save(payload, self.last_path)


def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
