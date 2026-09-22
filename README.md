# Pneumonia Classification with Explainable AI

Chest X-ray classifier using **PyTorch** + **DenseNet-121** with **Grad-CAM** attribution maps.

## Results

| Metric | Value |
|--------|-------|
| Test AUC-ROC | **0.971** |
| Sensitivity | **0.967** |
| Specificity | — |
| Test Set Size | 650 images |

> **Not for clinical use.** Research and educational purposes only.

## Architecture

- **Backbone:** DenseNet-121 (ImageNet pretrained)
- **Head:** Dropout(0.3) → Linear(1024 → 2)
- **Training:** Two-phase transfer learning (frozen → full fine-tune)
- **Imbalance handling:** WeightedRandomSampler + class-weighted CrossEntropyLoss
- **Threshold:** Youden's J statistic for optimal sensitivity/specificity trade-off

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-xai.git
cd pneumonia-xai
pip install -r requirements.txt

# Download the Kaggle dataset
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Place it under ./data/chest_xray/

python src/train.py        # trains and saves best model
python src/evaluate.py     # test metrics + ROC + confusion matrix
python src/gradcam.py      # Grad-CAM grid
```

## Grad-CAM

Heatmaps are generated using `pytorch-grad-cam` targeting `features.norm5` of DenseNet-121. Correctly classified pneumonia cases show warm activation in lower/perihilar lung zones; failure cases often show attention outside lung fields.

## Dataset

Kaggle Chest X-Ray Images (Pneumonia) — 5,863 pediatric anterior-posterior X-rays, two classes (`NORMAL`, `PNEUMONIA`).

## Repo Structure

```text
src/
├── dataset.py     # Data loading + weighted sampler
├── model.py       # DenseNet-121 wrapper
├── train.py       # Two-phase training
├── evaluate.py    # Clinical metrics
└── gradcam.py     # Attribution maps
```
