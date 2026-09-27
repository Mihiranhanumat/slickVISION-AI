# 🛰️ slickVISION-AI: Oil Spill Detection from SAR Satellite Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Chunk 1 Ready](https://img.shields.io/badge/Status-Chunk%201%20Ready-success.svg)](#-chunk-1-status)

**slickVISION-AI** is a deep learning and classical machine learning research pipeline designed for **5-class pixel-level semantic segmentation** of Sentinel-1 Synthetic Aperture Radar (SAR) imagery to detect marine oil spills, distinguish true oil slicks from confounding look-alikes, and identify vessels and coastlines.

---

## 🌊 Scientific Objectives

Synthetic Aperture Radar (SAR) provides active microwave ocean surface sensing through cloud cover and night conditions. Oil dampens ocean capillary waves and produces dark backscatter regions. However, natural low-wind areas, biogenic films, and meteorological conditions produce visually similar dark "look-alikes". 

This system solves the central scientific challenge:
> **Can a deep encoder-decoder segmentation model distinguish true oil slicks from visually similar SAR look-alikes while preserving small objects such as ships?**

### 🎨 Semantic Class Mapping

| ID | Class Name | Description | Visualization Color |
| :---: | :--- | :--- | :---: |
| **0** | **Sea Surface** | Background ocean backscatter | `[0, 0, 0]` (Black) |
| **1** | **Oil Spill** | Primary target: dampened capillary waves | `[0, 255, 255]` (Cyan) |
| **2** | **Look-alike** | Confounding non-oil dark phenomena (biogenic/low-wind) | `[255, 0, 0]` (Red) |
| **3** | **Ship** | Metallic vessels / corner reflectors | `[150, 0, 150]` (Purple) |
| **4** | **Land** | Coastal terrain and landmasses | `[0, 100, 0]` (Dark Green) |

---

## 📂 Repository Structure

```
slickVISION-AI/
├── README.md                      # Project overview and instructions
├── progress.md                    # Detailed Chunk 1-3 progress & roadmap
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment configuration template
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Package configuration
├── configs/
│   ├── base.yaml                  # Core configuration (paths, classes, patching, RF)
│   ├── unet_resnet34.yaml         # Chunk 2: U-Net ResNet34 configuration
│   └── deeplabv3plus.yaml         # Chunk 2: DeepLabV3+ configuration
├── data/
│   ├── raw/                       # Original SAR scenes (not committed)
│   ├── processed/                 # Preprocessed full scenes
│   ├── patches/                   # Extracted 256x256 training/val/test patches
│   └── splits/                    # Parent image lists, manifests, class weights
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Dataset inspection & class balance
│   └── 02_patch_visualization.ipynb # Patch extraction and augmentations
├── src/
│   └── oilspill/
│       ├── data/
│       │   ├── dataset.py         # Krestenitis dataset adapter & RGB mask converter
│       │   ├── patching.py        # 256x256 sliding window patch generator
│       │   ├── transforms.py      # Albumentations SAR-safe augmentations
│       │   └── validation.py      # Dataset auditor, class statistics & leakage checker
│       ├── baseline/
│       │   └── random_forest.py   # Multi-scale SAR feature engineering & Random Forest
│       ├── evaluation/
│       │   ├── metrics.py         # IoU, Dice/F1, Precision, Recall, mIoU calculations
│       │   └── confusion.py       # Confusion matrix plotting & look-alike error analysis
│       └── utils/
│           ├── seed.py            # Deterministic seeding
│           ├── io.py              # File readers/writers (YAML, JSON, images)
│           └── logging.py         # Structured logging
├── scripts/
│   ├── download_data.py           # Dataset acquisition instructions & synthetic generator
│   ├── generate_synthetic_data.py # Benchmark SAR scene generator with speckle physics
│   ├── prepare_data.py            # Master single-command data preparation
│   ├── audit_dataset.py           # Dataset statistics & Visual QA contact sheet
│   └── train_baseline.py          # Random Forest baseline training & evaluation
├── tests/
│   ├── test_dataset.py            # Dataset loader and mask integrity tests
│   ├── test_patching.py           # Patch extraction geometry tests
│   ├── test_split_leakage.py      # Zero-leakage parent scene verification
│   ├── test_transforms.py         # Discrete label augmentation tests
│   ├── test_metrics.py            # Segmentation metrics tests
│   └── test_baseline.py           # Random Forest baseline tests
└── artifacts/
    ├── checkpoints/               # Trained models (.joblib, .pt)
    ├── predictions/               # Prediction mask comparisons
    └── reports/                   # Audit reports, metrics JSON/CSV, QA sheets
```

---

## 📥 Dataset Acquisition

The primary benchmark is the **Krestenitis et al. (2019)** Oil Spill Detection Dataset consisting of 1,112 Sentinel-1 SAR scenes (1,002 train / 110 test, 1250×650 resolution).

### Accessing Official Dataset:
1. Visit the **CERTH M4D** portal: [https://m4d.iti.gr/oil-spill-detection-dataset/](https://m4d.iti.gr/oil-spill-detection-dataset/)
2. Request access using an institutional/academic email address or contact `mikrestenitis@iti.gr` / `kioannid@iti.gr`.
3. Extract the downloaded dataset into `data/raw/`.

### Local Synthetic Benchmark Generation (Instant Start):
If you do not have immediate access to the raw dataset, you can generate a high-fidelity synthetic benchmark with realistic SAR speckle physics and 5-class masks:
```bash
python scripts/download_data.py --synthetic
```

---

## ⚡ Quickstart & Running Chunk 1

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Single-Command Dataset Preparation
Splits parent training scenes into Train/Validation (80/20), keeps official Test scenes untouched, extracts 256×256 patches with overlap, and computes class weights:
```bash
python scripts/prepare_data.py --auto-generate
```

### 3. Generate Dataset Audit & Visual QA Contact Sheet
```bash
python scripts/audit_dataset.py
```
Outputs:
- Markdown report: `artifacts/reports/dataset_audit_report.md`
- Visual QA contact sheet: `artifacts/reports/visual_qa_contact_sheet.png`

### 4. Train & Evaluate Classical Baseline (Random Forest)
Extracts multi-scale local texture, window mean/std, and Sobel gradient features, trains a balanced Random Forest classifier, and evaluates against held-out images:
```bash
python scripts/train_baseline.py
```
Outputs:
- Checkpoint: `artifacts/checkpoints/random_forest_baseline.joblib`
- Metrics: `artifacts/reports/baseline_rf_metrics.json` and `.csv`
- Confusion Matrix: `artifacts/reports/baseline_rf_confusion_matrix.png`
- Qualitative Overlays: `artifacts/predictions/rf/`

### 5. Run Unit Tests
```bash
pytest
```

---

## 🧭 Next Steps (Chunk 2 & Chunk 3)

- **Chunk 2 (Deep Segmentation Training + Experiments):** Implementation of U-Net with ResNet34 encoder, DeepLabV3+, Weighted Cross-Entropy + Dice Loss, AMP training engine, and experiment tracking.
- **Chunk 3 (Inference Product + API + Frontend):** Tiled inference engine, FastAPI service, interactive React/Tailwind frontend, and Docker deployment.

See [progress.md](progress.md) for the complete roadmap and component breakdown.
