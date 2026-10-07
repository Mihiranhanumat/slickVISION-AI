# slickVISION-AI Reconstruction — Phase 1

## Purpose

This package is the first step of the reconstruction strategy aimed at substantially improving the
oil-spill segmentation score.

It intentionally does **not** start another expensive training run.

The current project has an important ambiguity:
- the methodology/presentation describes a 5-class task;
- the existing Chunk 2 plan describes the available 22,152-patch training data as binary.

We need to verify the actual mask IDs before choosing the 75%+ optimization route.

## Run

Open CMD in the project root:

```bat
python tools\audit_labels_and_data.py
```

For a full audit of every mask (slower):

```bat
python tools\audit_labels_and_data.py --sample-per-split 0
```

## What to send back

Send the terminal output.

The result will tell us whether the next reconstruction should be:

### Route A — binary
1. External oil-spill pretraining
2. Refined SOS-style supervised oil learning
3. DARTIS/CSIRO-style hard negatives
4. DeepLabV3+ ResNet-50 or stronger U-Net
5. Fine-tune on the project dataset
6. Optional probability ensemble / TTA
7. Final frozen test evaluation

### Route B — five-class
1. Correct 5-class model head
2. Class-aware sampling
3. Controlled class weights
4. CE + Dice / Tversky-style objective
5. Stronger architecture
6. Overlap-tile inference
7. Final frozen test evaluation

## Safety

Do not rerun `prepare_data.py`.
Do not modify the test split.
Do not tune using the official test results.
Do not claim 75% until the exact metric is measured on the frozen held-out test set.
