"""Data download and acquisition helper for Krestenitis SAR Oil Spill Dataset."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from oilspill.utils.logging import get_logger

logger = get_logger("DownloadData")

CERTH_M4D_URL = "https://m4d.iti.gr/oil-spill-detection-dataset/"
CONTACT_EMAILS = ["mikrestenitis@iti.gr", "kioannid@iti.gr"]


def check_existing_dataset(raw_dir: Path) -> bool:
    """Check if raw dataset files are already placed in data/raw."""
    if not raw_dir.exists():
        return False
    
    # Check for image and mask files
    img_files = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.bmp"))
    if len(img_files) > 0:
        logger.info(f"Found {len(img_files)} existing image/mask files in {raw_dir}")
        return True
    return False


def print_access_instructions() -> None:
    """Print detailed official instructions on obtaining the Krestenitis dataset."""
    print("=" * 80)
    print("OFFICIAL KRESTENITIS ET AL. (2019) SAR DATASET ACCESS INSTRUCTIONS")
    print("=" * 80)
    print(f"The official Krestenitis SAR Oil Spill dataset is hosted by CERTH/M4D:")
    print(f"Portal: {CERTH_M4D_URL}")
    print(f"Contacts: {', '.join(CONTACT_EMAILS)}")
    print()
    print("Step-by-step Access Guide:")
    print("1. Visit the CERTH M4D page and complete the dataset request form.")
    print("2. Submit using an institutional / university email address.")
    print("3. Once received, extract the archive into: slickVISION-AI/data/raw/")
    print("   Expected Structure:")
    print("   data/raw/")
    print("   ├── train/")
    print("   │   ├── images/   (1,002 SAR images, 1250x650)")
    print("   │   └── labels/   (1,002 5-class masks)")
    print("   └── test/")
    print("       ├── images/   (110 SAR images, 1250x650)")
    print("       └── labels/   (110 5-class masks)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Acquire or generate benchmark SAR dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Target raw data directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic Sentinel-1 SAR benchmark scenes")
    parser.add_argument("--num-synthetic-train", type=int, default=10, help="Number of synthetic train scenes")
    parser.add_argument("--num-synthetic-test", type=int, default=4, help="Number of synthetic test scenes")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    if check_existing_dataset(raw_dir):
        logger.info(f"Raw dataset is ready in {raw_dir}.")
        return

    print_access_instructions()

    if args.synthetic:
        logger.info("Generating synthetic SAR benchmark dataset for development & testing...")
        from generate_synthetic_data import generate_benchmark
        generate_benchmark(
            output_dir=raw_dir,
            num_train=args.num_synthetic_train,
            num_test=args.num_synthetic_test,
        )
        logger.info(f"Synthetic benchmark successfully generated in {raw_dir}!")
    else:
        logger.info(
            "To generate a local development SAR benchmark immediately, run:\n"
            "   python scripts/download_data.py --synthetic"
        )


if __name__ == "__main__":
    main()
