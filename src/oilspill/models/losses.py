"""Loss functions for imbalanced SAR segmentation.

All losses take logits (B, C, H, W) and integer targets (B, H, W).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()


class CombinedCEDiceLoss(nn.Module):
    """L = ce_weight * weighted_CE + dice_weight * soft_Dice  (lambda = 0.5 by default).

    Weighted CE punishes mistakes on rare classes more; soft Dice rewards region overlap directly.
    Dice is computed per class over the whole batch and averaged across classes.
    """

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None, smooth: float = 1.0):
        super().__init__()
        self.ce_weight, self.dice_weight, self.smooth = ce_weight, dice_weight, smooth
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = F.cross_entropy(logits, targets.long(), weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)
        onehot = _one_hot(targets, logits.shape[1])
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice.mean())


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, 2019), computed on foreground classes (1..C-1).

    TI = TP / (TP + alpha*FN + beta*FP);  loss = mean_c (1 - TI_c) ** gamma.
    alpha > beta penalises missed oil (false negatives) more than false alarms.
    An optional small CE term (ce_weight) can stabilise early training.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75,
                 smooth: float = 1.0, ce_weight: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth, self.ce_weight = alpha, beta, gamma, smooth, ce_weight
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        probs = torch.softmax(logits, dim=1)
        onehot = _one_hot(targets, logits.shape[1])
        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss = ((1.0 - ti[1:]) ** self.gamma).mean()
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * F.cross_entropy(logits, targets.long(), weight=self.class_weights)
        return loss


def build_loss(cfg: Dict[str, Any], class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Create the loss named in cfg['training']['loss']['name'].

    Supported: 'ce', 'combined_ce_dice', 'focal_tversky'.
    Class weights are only used when loss.use_class_weights is true.
    """
    lc = cfg["training"]["loss"]
    name = lc.get("name", "combined_ce_dice").lower()
    w = class_weights if lc.get("use_class_weights", False) else None
    if name == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if name == "combined_ce_dice":
        return CombinedCEDiceLoss(lc.get("ce_weight", 0.5), lc.get("dice_weight", 0.5), w)
    if name == "focal_tversky":
        return FocalTverskyLoss(lc.get("alpha", 0.7), lc.get("beta", 0.3), lc.get("gamma", 0.75),
                                ce_weight=lc.get("ft_ce_weight", 0.0),
                                class_weights=w)
    raise ValueError(f"Unknown loss '{name}'")
