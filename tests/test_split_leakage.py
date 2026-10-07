"""Unit test ensuring zero data leakage across train, val, and test splits."""

import pytest
from pathlib import Path


def test_split_leakage_manifest_isolation():
    """Verify that split lists never share parent images."""
    splits_dir = Path("data/splits")
    if not (splits_dir / "train_images.txt").exists():
        pytest.skip("Splits directory not prepared yet.")

    with open(splits_dir / "train_images.txt") as f:
        train_parents = set(line.strip() for line in f if line.strip())

    with open(splits_dir / "val_images.txt") as f:
        val_parents = set(line.strip() for line in f if line.strip())

    with open(splits_dir / "test_images.txt") as f:
        test_parents = set(line.strip() for line in f if line.strip())

    # Strict disjointness check
    train_val_overlap = train_parents.intersection(val_parents)
    train_test_overlap = train_parents.intersection(test_parents)
    val_test_overlap = val_parents.intersection(test_parents)

    assert len(train_val_overlap) == 0, f"Data leakage detected between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Data leakage detected between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage detected between Val and Test: {val_test_overlap}"
