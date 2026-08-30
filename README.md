# Lung Cancer CT Classification

Deep learning system for classifying lung CT scans into **normal / benign / malignant**,
based on a dual-backbone (ResNet50 + EfficientNetB3) feature-fusion architecture,
with SHAP-based explainability for tumor localization.

Based on: *Comparative Evaluation of ResNet50, Xception, and EfficientNetB3 for
Lung Cancer Classification and Explainable AI-based Tumor Localization*
(Sharma, Srivats, P B, R, R — VIT Chennai).

## Status
🚧 Phase 0 — repo scaffolding. Model, training, and API not yet implemented.

## Project structure

```
src/
  config.py        # all hyperparameters and paths
  data_loader.py    # CT scan loading, preprocessing, splitting
  model.py          # dual-branch fusion architecture
  train.py          # training loop
  inference.py      # prediction on new images
  explain.py         # SHAP heatmap generation
tests/               # unit tests
data/                # gitignored — place raw CT scans in data/raw/
models/              # gitignored — trained model artifacts
.github/workflows/    # CI/CD pipeline (added Phase 7)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Architecture

```
Input image
    |
  +-+-------------------+
  |                       |
ResNet50            EfficientNetB3
  |                       |
Flatten               Flatten
  |                       |
  +---------+-------------+
       Concatenate
            |
    Dense(128, ReLU)
            |
       Dense(3)
            |
        Softmax
```
