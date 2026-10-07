# slickVISION-AI — Experimental Results

## Project

SAR oil-spill semantic segmentation using DeepLab-based segmentation models.

## Best Source-Domain Model

Architecture:
- DeepLabV3+ with ResNet50 encoder
- Binary segmentation
- Class 0: Background
- Class 1: Oil Spill

Dataset:
- Refined SOS
- Training: 6,455 images
- Validation: 1,615 images

Best validation checkpoint:

`artifacts/rescue_checkpoints/stage1_20epoch/refined_sos_best.pth`

### Refined SOS Validation Results

| Metric | Result |
|---|---:|
| Oil IoU | **74.50%** |
| Dice / F1 | **85.39%** |
| Precision | **84.63%** |
| Recall | **86.16%** |

These metrics represent the Refined SOS validation set and are not the final CSIRO test score.

## Cross-Domain Transfer

The Refined SOS-trained checkpoint was evaluated directly on the project's CSIRO test split.

### CSIRO Test — Zero-Shot Transfer

| Metric | Result |
|---|---:|
| Oil IoU | **9.74%** |
| Dice / F1 | **17.75%** |
| Precision | **12.75%** |
| Recall | **29.22%** |
| Pixel Accuracy | **81.03%** |

This result demonstrates a substantial domain-transfer gap between Refined SOS and the project's CSIRO distribution.

## Qualitative Results

Publication-ready qualitative examples are stored under:

`artifacts/paper_figures/`

Each figure contains:

1. SAR input
2. Ground-truth oil mask
3. Oil probability map
4. Predicted segmentation mask
5. Prediction overlay

These figures are qualitative examples from the Refined SOS validation set.

## Interpretation

The experiments demonstrate that the model learns strong oil-spill segmentation on the Refined SOS source domain, achieving 74.50% Oil IoU and 85.39% Dice.

However, direct transfer to the project's CSIRO test distribution results in substantially lower performance, indicating significant domain shift.

The next research direction is target-domain adaptation and multi-source hard-negative training.

## Reference Benchmark

The public DeepLabV3-ResNet50 oil-spill workflow that motivated the reconstruction reports 77.69% IoU on the Refined SOS validation protocol after hard-negative fine-tuning with DARTIS and CSIRO no-oil samples.

That reported value should not be interpreted as the CSIRO test performance of this project.