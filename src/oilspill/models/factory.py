"""Segmentation model factory built on segmentation_models_pytorch (smp)."""

from typing import Any, Dict

import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger("ModelFactory")

ARCHITECTURES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
    "fpn": "FPN",
}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build a segmentation model from a (merged) config dictionary.

    Reads cfg["model"]: architecture, encoder_name, encoder_weights, in_channels, classes, activation.
    SAR is single-channel, so in_channels=1 is used directly: smp adapts the first convolution of
    the ImageNet-pretrained encoder (it sums the RGB filters), so no channel replication is needed.
    """
    import segmentation_models_pytorch as smp

    m = cfg["model"]
    arch = str(m.get("architecture", "unet")).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {sorted(ARCHITECTURES)}")
    cls = getattr(smp, ARCHITECTURES[arch])

    kwargs = dict(
        encoder_name=m.get("encoder_name", "resnet34"),
        encoder_weights=m.get("encoder_weights", "imagenet"),
        in_channels=int(m.get("in_channels", 1)),
        classes=int(m.get("classes", 2)),
        activation=m.get("activation", None),
    )
    try:
        model = cls(**kwargs)
        used = kwargs["encoder_weights"]
    except Exception as exc:  # most often: no internet to download pretrained weights
        if kwargs["encoder_weights"] is None:
            raise
        logger.warning(
            f"Could not load pretrained '{kwargs['encoder_weights']}' weights ({exc!s:.120}). "
            "Falling back to RANDOM initialisation. Check your internet connection if unintended."
        )
        kwargs["encoder_weights"] = None
        model = cls(**kwargs)
        used = None
    model.encoder_weights_used = used  # recorded in checkpoints / reports
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
