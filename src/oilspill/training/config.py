"""Config loading with inheritance (`inherits:` / `base_config:`) and dotted CLI overrides."""

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from ..utils.io import load_yaml


def deep_merge(base: Dict[str, Any], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply overrides such as 'training.epochs=5' or 'model.architecture=deeplabv3plus'."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving `base_config` and `inherits` parents recursively."""
    raw = load_yaml(path)
    base_path = raw.pop("base_config", None)
    parent_path = raw.pop("inherits", None)
    cfg: Dict[str, Any] = {}
    if base_path:
        cfg = deep_merge(cfg, load_yaml(base_path))
    if parent_path:
        cfg = deep_merge(cfg, load_config(parent_path))
    cfg = deep_merge(cfg, raw)
    return apply_overrides(cfg, overrides)
