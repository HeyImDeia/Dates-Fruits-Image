# Date Fruit Classification

Classifies 9 varieties of date fruits from images using traditional computer vision and ML.

## Dataset

1,658 images across 9 classes (Ajwa, Galaxy, Medjool, Meneifi, Nabtat Ali, Rutab, Shaishe, Sokari, Sugaey) — sourced from [Kaggle](https://www.kaggle.com), originally from Alhamdan & Howe, 2021.

## Pipeline

1. **Preprocessing** — resize to 224×224, HEIC conversion
2. **Segmentation** — GrabCut background removal
3. **Enhancement** — CLAHE contrast boost (LAB color space)
4. **Feature extraction** — 256-dim vector (192 color + 64 Sobel gradient)
5. **Classification** — KNN, SVM, Random Forest, Logistic Regression, MLP

## Results

| Model | Accuracy |
|---|---|
| Cosine 5-NN | 72.59% |
| KNN | 79.22% |
| Random Forest | 84.94% |
| SVM | 86.14% |
| Logistic Regression | 87.35% |
| **MLP** | **89.16%** |

## Stack

Python · scikit-learn · OpenCV · Pillow · NumPy · Matplotlib
