"""I/O helper utilities for YAML, JSON, CSV, and image files."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        path: Path to YAML file.
        
    Returns:
        Dictionary containing configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to a YAML file.
    
    Args:
        data: Dictionary data.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        Parsed JSON dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to serialize.
        path: Destination path.
        indent: Indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def default_converter(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=default_converter)


def load_image(path: Union[str, Path], grayscale: bool = True) -> np.ndarray:
    """Load an image using OpenCV with fallback to PIL.
    
    Args:
        path: Path to image file.
        grayscale: If True, load as 1-channel grayscale image.
        
    Returns:
        np.ndarray image array (H, W) or (H, W, C).
    """
    path = str(path)
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    if img is None:
        raise ValueError(f"Could not load image at path: {path}")
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save an image array to disk.
    
    Args:
        img: np.ndarray image.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    else:
        cv2.imwrite(str(path), img)
