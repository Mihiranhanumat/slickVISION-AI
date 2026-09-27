# 📊 slickVISION-AI Project Progress & Architecture Roadmap

This document outlines the implementation status of **slickVISION-AI** (Deep Learning Semantic Segmentation for Oil Spill Detection from Sentinel-1 SAR Imagery), following the **Antigravity 3-Chunk Build Blueprint**.

---

## 🗺️ High-Level 3-Chunk Project Matrix

| Chunk | Phase / Scope | Target Deliverables | Current Status | Responsible Team |
| :--- | :--- | :--- | :---: | :--- |
| **Chunk 1** | **Data Engine + Baseline + Skeleton** | Data loaders, audit, leakage-free split, 256×256 patching, augmentations, Random Forest baseline, test suite | **✅ COMPLETED** | Completed in this sprint |
| **Chunk 2** | **Deep Segmentation + Experiments** | U-Net + ResNet34, DeepLabV3+, Weighted CE + Dice, AMP training engine, experiment matrix (E0–E4) | **⏳ PENDING** | Teammate Sprint (Chunk 2) |
| **Chunk 3** | **Inference + API + UI + Packaging** | Tiled inference stitching, FastAPI backend, React/Tailwind frontend, Docker Compose, final report | **⏳ PENDING** | Teammate Sprint (Chunk 3) |

---

## ✅ Chunk 1: What is Present & Completed

### 1. Project Skeleton & Configuration
- [x] **Modular Structure:** Clean Python package layout under `src/oilspill/` with separation of concerns (`data/`, `baseline/`, `evaluation/`, `utils/`).
- [x] **Config Management:** Centralized YAML configuration files (`configs/base.yaml`, `configs/unet_resnet34.yaml`, `configs/deeplabv3plus.yaml`).
- [x] **Environment & Build:** `.env.example`, `.gitignore`, `requirements.txt`, `pyproject.toml`, and MIT `LICENSE`.

### 2. Dataset Adapter & Label Integrity
- [x] **5-Class Semantic Labeling:** Strict enforcement of class IDs `[0, 4]` (0: Sea surface, 1: Oil spill, 2: Look-alike, 3: Ship, 4: Land).
- [x] **RGB Mask Decoder & Reversibility:** Nearest-color Euclidean mapping (`rgb_to_mask`) to handle standard Krestenitis color palettes and compression artifacts, plus `mask_to_rgb` for visual rendering.
- [x] **Dataset Adapter:** `KrestenitisDataset` with lazy reading and Albumentations integration.

### 3. Dataset Audit & Preprocessing
- [x] **Dataset Auditor:** `DatasetAuditor` checks pair counts, image/mask dimensions (1250×650), detects invalid labels, calculates class pixel distributions, and determines Median Frequency class weights.
- [x] **Automated Reports:** Generates `artifacts/reports/dataset_audit_report.json` and `dataset_audit_report.md`.
- [x] **Visual QA Contact Sheet:** `scripts/audit_dataset.py` generates multi-sample contact sheets with SAR scene, ground truth mask, overlay, and unified legend.

### 4. Zero-Leakage Train/Validation/Test Split Engine
- [x] **Parent-Image Level Splitting:** Strict separation of parent scenes before patch extraction (80% train, 20% validation). Official test scenes remain untouched.
- [x] **Split Manifests:** Saves `data/splits/train_images.txt`, `data/splits/val_images.txt`, `data/splits/test_images.txt`.
- [x] **No Leakage Guarantee:** Test-set and validation statistics are strictly isolated from training normalization.

### 5. 256×256 Patch Extractor & Manifest
- [x] **Sliding Window Tiling:** `PatchExtractor` generates 256×256 patches with configurable overlap (64px overlap for train stride 192; deterministic 0px overlap for validation/testing).
- [x] **Patch Manifest:** Builds `data/splits/patch_manifest.csv` with per-patch oil pixel counts, bounding coordinates, parent image lineage, and split assignments.

### 6. Augmentation Pipeline
- [x] **SAR-Safe Transforms:** `transforms.py` provides spatial transforms (HorizontalFlip, VerticalFlip, RandomRotate90) and mild intensity adjustments that preserve SAR physics. Nearest-neighbor interpolation prevents mask label distortion.

### 7. Classical Baseline (Random Forest)
- [x] **SAR Feature Engineering:** Dense multi-scale pixel feature extraction (raw intensity, window means & standard deviations across 3×3, 7×7, 15×15, Sobel gradients dx/dy/magnitude, and Laplacian response).
- [x] **Balanced Sampling:** Stratified pixel extraction so majority sea pixels do not overpower minority oil and look-alike pixels.
- [x] **Evaluation & Artifacts:** Full validation evaluation producing `artifacts/reports/baseline_rf_metrics.json`, `baseline_rf_metrics.csv`, `baseline_rf_confusion_matrix.png`, and qualitative overlay plots in `artifacts/predictions/rf/`.

### 8. Full Unit Test Suite & Notebooks
- [x] **Unit Tests:** `pytest` test suite covering dataset shapes, RGB-to-mask conversion, patch coordinates, zero data leakage, augmentation label preservation, metrics calculation, and RF baseline fitting.
- [x] **Jupyter Notebooks:** `notebooks/01_data_exploration.ipynb` and `notebooks/02_patch_visualization.ipynb`.

---

## 🚫 Chunk 2: What Should NOT Be in Chunk 1 (Reserved for Teammates)

The following components are intentionally left for Chunk 2 to maintain modularity:
1. **Deep Learning Model Factory:** PyTorch U-Net with ResNet34 encoder and DeepLabV3+ model classes (`src/oilspill/models/factory.py`).
2. **Loss Functions:** Weighted Cross-Entropy + Dice Loss / Focal Tversky Loss implementation (`src/oilspill/models/losses.py`).
3. **Deep Training Engine:** PyTorch AMP (Automatic Mixed Precision), AdamW optimizer, CosineAnnealingLR scheduler, early stopping, and checkpoint management (`src/oilspill/training/train.py`, `engine.py`).
4. **Experiment Tracking:** MLflow / WandB run logging and model comparison experiments (E0–E4).

---

## 🚫 Chunk 3: What Should NOT Be in Chunk 1 (Reserved for Teammates)

The following components are intentionally left for Chunk 3:
1. **Full-Scene Tiled Inference Engine:** Sliding-window patch prediction, overlap blending/stitching, and confidence map generation (`src/oilspill/inference/predictor.py`, `tiling.py`).
2. **FastAPI Backend Service:** REST endpoints (`/health`, `/model-info`, `/predict`, `/predict/overlay`, `/predict/compare`).
3. **React + Tailwind Frontend UI:** Interactive web dashboard for SAR image upload, side-by-side mask visualization, confidence heatmaps, and error analysis comparison screens.
4. **Containerization & Deployment:** Production `Dockerfile` and `docker-compose.yml`.

---

## 🚀 How Teammates Can Run & Test Chunk 1

### 1. Setup & Environment
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data (Audit + Split + Patches + Manifest)
```bash
# If raw Krestenitis dataset is placed in data/raw/:
python scripts/prepare_data.py

# If testing with synthetic benchmark scenes:
python scripts/prepare_data.py --auto-generate
```

### 3. Generate Visual QA Contact Sheet & Audit
```bash
python scripts/audit_dataset.py
```

### 4. Train & Evaluate Classical Random Forest Baseline
```bash
python scripts/train_baseline.py
```

### 5. Run All Unit Tests
```bash
pytest
```
