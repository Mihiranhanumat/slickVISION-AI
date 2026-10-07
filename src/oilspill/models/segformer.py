from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerSAR(nn.Module):
    """SegFormer wrapper for 1-channel SAR input and binary segmentation."""
    def __init__(self, pretrained_name="nvidia/mit-b2", num_classes=2, in_channels=1):
        super().__init__()
        if in_channels != 1:
            raise ValueError("This wrapper expects 1-channel SAR input.")
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("Install with: pip install transformers") from exc
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("Expected 1 or 3 input channels.")
        size = x.shape[-2:]
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
