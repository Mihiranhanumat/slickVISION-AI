from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerSAR5Class(nn.Module):
    """SegFormer MiT-B2 adapted to 1-channel SAR and 5-class segmentation."""

    def __init__(
        self,
        pretrained_name: str = "nvidia/mit-b2",
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        if num_classes != 5:
            raise ValueError("This project configuration expects exactly 5 classes.")

        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Run: pip install transformers"
            ) from exc

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input [B,1,H,W], got {tuple(x.shape)}"
            )

        original_size = x.shape[-2:]
        x = x.float()

        # Patches are expected to be in [0,1]. Protect against uint8-range data.
        if torch.any(x < 0.0) or torch.any(x > 1.0):
            x = torch.clamp(x / 255.0, 0.0, 1.0)

        # Keep SAR intensity identical across the three input channels.
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std

        logits = self.model(pixel_values=x).logits

        return F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
