# ?? slickVISION-AI Project Progress & Architecture Roadmap

This document outlines the implementation status of **slickVISION-AI** (Deep Learning Semantic Segmentation for Oil Spill Detection from Sentinel-1 SAR Imagery), following the **Antigravity 3-Chunk Build Blueprint**.

---

## ??? High-Level 3-Chunk Project Matrix

| Chunk | Phase / Scope | Target Deliverables | Current Status | Responsible Team |
| :--- | :--- | :--- | :---: | :--- |
| **Chunk 1** | **Data Engine + Baseline + Skeleton** | Real SAR data ingestion, audit, leakage-free split, 400x400 patching, augmentations, Random Forest baseline, test suite | **COMPLETED** | Completed in this sprint |
| **Chunk 2** | **Deep Segmentation + Experiments** | U-Net + ResNet34, DeepLabV3+, Weighted CE + Dice, AMP training engine, experiment matrix (E0-E4) | **PENDING** | Teammate Sprint (Chunk 2) |
| **Chunk 3** | **Inference + API + UI + Packaging** | Tiled inference stitching, FastAPI backend, React/Tailwind frontend, Docker Compose, final report | **PENDING** | Teammate Sprint (Chunk 3) |

---

## Chunk 1: What is Present & Completed

### 1. Project Skeleton & Configuration
- [x] Modular Structure: Clean Python package layout under src/oilspill/ with separation of concerns.
- [x] Config Management: Centralized YAML configuration files (configs/base.yaml, configs/unet_resnet34.yaml, configs/deeplabv3plus.yaml).
- [x] Environment & Build: .env.example, .gitignore, requirements.txt, pyproject.toml, and MIT LICENSE.

### 2. Real Dataset: Krestenitis et al. (2019) / CSIRO Sentinel-1 SAR Chips
- [x] Source: Krestenitis et al. (2019) -- Sentinel-1 SAR chips, CSIRO collection.
- [x] Total Chips: 5,630 images at 400x400 pixels (3,725 Class_0 / no oil spill; 1,905 Class_1 / oil spill).
- [x] Data Ingestion Module: src/oilspill/data/ingest.py -- reads real JPEG SAR chips, generates binary segmentation masks using Otsu thresholding.
- [x] No Synthetic Data: The pipeline exclusively uses the real dataset. Synthetic generation removed.
- [x] Dataset Metadata Committed: data/kaggle/metadata/ (XML metadata, readme, licence) committed to Git.
- [x] Raw Images Excluded from Git: data/kaggle/data/ is in .gitignore -- images remain local.

### 3. Dataset Audit & Preprocessing
- [x] DatasetAuditor checks pair counts, image dimensions, invalid labels, class pixel distributions, Median Frequency class weights.
- [x] Generates artifacts/reports/dataset_audit_report.json and .md.
- [x] scripts/audit_dataset.py generates visual QA contact sheets.

### 4. Zero-Leakage Train/Validation/Test Split Engine
- [x] Parent-Image Level Splitting: Strict separation before patch extraction (80% train, 10% val, 10% test).
- [x] Split Manifests: data/splits/train_images.txt, val_images.txt, test_images.txt.
- [x] No Leakage Guarantee: Validation and test statistics isolated from training normalization.

### 5. 400x400 Patch Extractor & Manifest

Patch Statistics (as of last run):

| Split | Patches | Oil-Spill Patches |
|:------|--------:|------------------:|
| Train | 16,000  | ~4,750            |
| Val   | 2,828   | ~843              |
| Test  | 3,324   | ~1,000            |
| Total | 22,152  | 6,593 (29.8%)     |

- [x] Sliding Window Tiling with configurable overlap (stride=200 train; stride=400 val/test).
- [x] Patch Manifest: data/splits/patch_manifest.csv with per-patch metadata.

### 6. Augmentation Pipeline
- [x] SAR-Safe Transforms: HorizontalFlip, VerticalFlip, RandomRotate90 plus mild intensity adjustments.

### 7. Classical Baseline (Random Forest)
- [x] SAR Feature Engineering: intensity, window stats (3x3/7x7/15x15), Sobel, Laplacian.
- [x] Balanced Sampling: Stratified pixel extraction.
- [x] Evaluation artifacts: baseline_rf_metrics.json, confusion matrix PNG, overlay plots.

### 8. Unit Test Suite & Notebooks
- [x] pytest suite: dataset shapes, patch coords, zero leakage, augmentation, metrics, RF fitting.
- [x] Notebooks: 01_data_exploration.ipynb, 02_patch_visualization.ipynb.

---

## Repository Status

| Item | Status |
|:-----|:-------|
| Code pushed to origin/main | YES -- commit 75be801 |
| Dataset metadata committed | YES -- data/kaggle/metadata/ |
| Raw images committed | NO -- excluded via .gitignore (kept local at C:\Users\Admin\Downloads\DL) |
| Data pipeline verified on real data | YES -- 22,152 patches generated |

---

## Chunk 2: What Should NOT Be in Chunk 1 (Reserved for Teammates)

1. Deep Learning Model Factory: U-Net + ResNet34, DeepLabV3+ (src/oilspill/models/factory.py).
2. Loss Functions: Weighted CE + Dice / Focal Tversky (src/oilspill/models/losses.py).
3. Deep Training Engine: PyTorch AMP, AdamW, CosineAnnealingLR, early stopping, checkpoints.
4. Experiment Tracking: MLflow / WandB run logging and experiments (E0-E4).

---

## Chunk 3: What Should NOT Be in Chunk 1 (Reserved for Teammates)

1. Full-Scene Tiled Inference Engine: sliding-window prediction and stitching.
2. FastAPI Backend Service: /health, /predict, /predict/overlay, /predict/compare.
3. React + Tailwind Frontend UI: web dashboard for upload, visualization, heatmaps.
4. Containerization: Dockerfile and docker-compose.yml.

---

## How Teammates Can Run Chunk 1

Pre-requisite: Place raw SAR chips under data/kaggle/data/Class_0/ and data/kaggle/data/Class_1/
(or update configs/base.yaml data.kaggle_dir to your local path).

### 1. Setup
pip install -r requirements.txt
pip install -e .

### 2. Prepare Data
python scripts/prepare_data.py

### 3. Audit & Visual QA
python scripts/audit_dataset.py

### 4. Train RF Baseline
python scripts/train_baseline.py

### 5. Run Tests
pytest
