"""Synthetic Sentinel-1 SAR imagery and 5-class semantic mask generator.

Generates realistic benchmark scenes (1250x650) with SAR physics:
- Sea surface: Ocean backscatter with multiplicative speckle noise
- Oil spills: Capillary wave damping causing characteristic dark slicks
- Look-alikes: Low-wind zones and biogenic surface films
- Ships: Strong corner reflectors (high backscatter point targets)
- Land: Coastal landmasses with high roughness
"""

import argparse
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

from oilspill.data.dataset import mask_to_rgb
from oilspill.utils.logging import get_logger

logger = get_logger("SyntheticSARGenerator")


def create_synthetic_sar_scene(
    height: int = 650,
    width: int = 1250,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single SAR scene and corresponding 5-class ground truth mask.
    
    Returns:
        sar_image: (H, W) uint8 grayscale SAR image.
        mask: (H, W) uint8 mask with values in [0, 4].
    """
    rng = np.random.default_rng(seed)
    
    # Base background: Sea surface (Class 0)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 1. Coastal landmass (Class 4) on one edge (left, right, top, or bottom)
    if rng.random() > 0.3:
        land_edge = rng.choice(["left", "right", "bottom", "top"])
        land_pts = []
        if land_edge == "left":
            x_boundary = rng.integers(100, 300)
            pts = [[0, 0], [x_boundary, 0]]
            for y in range(50, height, 50):
                pts.append([int(x_boundary + rng.integers(-40, 40)), y])
            pts.extend([[x_boundary, height], [0, height]])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 4)
        elif land_edge == "right":
            x_boundary = width - rng.integers(100, 300)
            pts = [[width, 0], [x_boundary, 0]]
            for y in range(50, height, 50):
                pts.append([int(x_boundary + rng.integers(-40, 40)), y])
            pts.extend([[x_boundary, height], [width, height]])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 4)

    # 2. Look-alikes (Class 2): larger, diffuse dark areas (biogenic slicks / low wind)
    num_lookalikes = rng.integers(1, 4)
    for _ in range(num_lookalikes):
        cx, cy = rng.integers(100, width - 100), rng.integers(100, height - 100)
        axes = (rng.integers(60, 160), rng.integers(30, 90))
        angle = rng.integers(0, 180)
        temp = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(temp, (int(cx), int(cy)), axes, float(angle), 0, 360, 255, -1)
        # Apply only to ocean
        mask[(temp == 255) & (mask == 0)] = 2

    # 3. Oil spills (Class 1): elongated curved or linear streaks with sharp damping
    num_spills = rng.integers(1, 3)
    for _ in range(num_spills):
        start_x, start_y = rng.integers(150, width - 150), rng.integers(150, height - 150)
        pts = [[start_x, start_y]]
        cur_x, cur_y = start_x, start_y
        steps = rng.integers(6, 14)
        for _ in range(steps):
            cur_x += rng.integers(-35, 45)
            cur_y += rng.integers(-25, 35)
            pts.append([int(cur_x), int(cur_y)])
            
        temp = np.zeros((height, width), dtype=np.uint8)
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        thickness = rng.integers(15, 35)
        cv2.polylines(temp, [pts_arr], isClosed=False, color=255, thickness=thickness)
        mask[(temp == 255) & (mask == 0)] = 1

    # 4. Ships (Class 3): small bright localized signatures
    num_ships = rng.integers(1, 4)
    ship_locations = []
    for _ in range(num_ships):
        sx, sy = rng.integers(50, width - 50), rng.integers(50, height - 50)
        # Place ship on sea or near oil
        if mask[sy, sx] != 4:
            s_radius = rng.integers(2, 4)
            cv2.circle(mask, (int(sx), int(sy)), int(s_radius), 3, -1)
            ship_locations.append((sx, sy, s_radius))

    # --- Generate Synthetic SAR Intensities ---
    # Sea surface baseline backscatter (mean ~110 with multiplicative Rayleigh/Gamma speckle)
    sar_intensity = rng.gamma(shape=4.0, scale=28.0, size=(height, width)).astype(np.float32)

    # Land: high roughness and heterogeneous backscatter
    land_mask = (mask == 4)
    sar_intensity[land_mask] = rng.gamma(shape=7.0, scale=25.0, size=np.sum(land_mask))

    # Look-alikes: moderate damping (~65 intensity)
    lookalike_mask = (mask == 2)
    sar_intensity[lookalike_mask] = rng.gamma(shape=3.0, scale=20.0, size=np.sum(lookalike_mask))

    # Oil spills: severe wave damping (~35 intensity)
    oil_mask = (mask == 1)
    sar_intensity[oil_mask] = rng.gamma(shape=2.5, scale=12.0, size=np.sum(oil_mask))

    # Ships: strong corner reflector (>240 intensity)
    ship_mask = (mask == 3)
    sar_intensity[ship_mask] = np.clip(rng.normal(245.0, 10.0, size=np.sum(ship_mask)), 220, 255)

    # Clip to valid 8-bit range
    sar_image = np.clip(sar_intensity, 0, 255).astype(np.uint8)
    
    return sar_image, mask


def generate_benchmark(
    output_dir: Union[str, Path],
    num_train: int = 10,
    num_test: int = 4,
    height: int = 650,
    width: int = 1250,
    seed: int = 42,
) -> None:
    """Generate train and test benchmark datasets."""
    out_path = Path(output_dir)
    
    train_img_dir = out_path / "train" / "images"
    train_lbl_dir = out_path / "train" / "labels"
    train_vis_dir = out_path / "train" / "visual_masks"
    test_img_dir = out_path / "test" / "images"
    test_lbl_dir = out_path / "test" / "labels"
    test_vis_dir = out_path / "test" / "visual_masks"

    for d in [train_img_dir, train_lbl_dir, train_vis_dir, test_img_dir, test_lbl_dir, test_vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_train} training scenes and {num_test} testing scenes ({width}x{height})...")

    # Generate train scenes
    for i in range(num_train):
        img, mask = create_synthetic_sar_scene(height=height, width=width, seed=seed + i)
        stem = f"synthetic_sar_train_{i+1:04d}"
        
        cv2.imwrite(str(train_img_dir / f"{stem}.png"), img)
        cv2.imwrite(str(train_lbl_dir / f"{stem}.png"), mask)
        
        # Save RGB visual mask as well
        vis_mask = mask_to_rgb(mask)
        cv2.imwrite(str(train_vis_dir / f"{stem}.png"), cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))

    # Generate test scenes
    for j in range(num_test):
        img, mask = create_synthetic_sar_scene(height=height, width=width, seed=seed + 1000 + j)
        stem = f"synthetic_sar_test_{j+1:04d}"
        
        cv2.imwrite(str(test_img_dir / f"{stem}.png"), img)
        cv2.imwrite(str(test_lbl_dir / f"{stem}.png"), mask)
        
        vis_mask = mask_to_rgb(mask)
        cv2.imwrite(str(test_vis_dir / f"{stem}.png"), cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))

    logger.info(f"Successfully generated dataset at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SAR oil spill benchmark.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--train-scenes", type=int, default=10, help="Number of train scenes")
    parser.add_argument("--test-scenes", type=int, default=4, help="Number of test scenes")
    args = parser.parse_args()

    generate_benchmark(
        output_dir=args.output_dir,
        num_train=args.train_scenes,
        num_test=args.test_scenes,
    )
