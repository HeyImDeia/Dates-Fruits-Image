# Dates Fruit Image Classification — ARTI 404

## Project Overview
This project classifies 9 types of date fruits using image processing and computer vision.
Dataset: 1,658 high-quality JPG images across 9 classes, sourced from Kaggle (`archive.zip`).
Originally published in: *"Classification of Date Fruits in a Controlled Environment Using CNNs"* — Alhamdan & Howe, 2021 (Springer). DOI: 10.1007/978-3-030-69717-4_16

**Dataset imaging setup:** Canon EOS 550D DSLR, ring light (48cm, 240 LEDs, 100% brightness), white paper background, fixed 8cm camera-to-fruit distance. All images were taken in a controlled lab — background is consistently white across the entire dataset.

## Dataset Structure
- **Raw images:** `date_fruit_dataset/` — one subfolder per class
- **Classes (9):** Ajwa (175), Galaxy (190), Medjool (135), Meneifi (232), Nabtat Ali (177), Rutab (146), Shaishe (171), Sokari (264), Sugaey (168)
- **Processed images:** `processed/` — all images resized to 224×224 JPEG, named `ClassName_stem.jpg`
- **Segmented images:** `segmented/` — GrabCut background-removed versions of `processed/`
- **CLAHE images:** `clahe/` — contrast-enhanced versions of `processed/` (CLAHE on L channel in LAB space)
- **File types:** `.jpg`, `.jpeg`, `.png`, `.heic` — HEIC files are converted to `.jpg` in place
- **HEIC support:** requires `pillow-heif` — call `register_heif_opener()` before opening images

## Notebook Layout (`download_dataset.ipynb`)
| # | Cell ID | Type | Purpose |
|---|---------|------|---------|
| 0 | `cell-0` | code | pip installs — commented out, run once |
| 1 | `4c075d1b` | code | Colab setup — auto-detects Colab, mounts Drive, skipped locally |
| 2 | `5fae6f85` | markdown | Extract zip header |
| 3 | `716cb605` | markdown | EDA header |
| 4 | `5512690a` | code | Unzips `archive.zip` → `date_fruit_kaggle/` |
| 5 | `db5e468c` | code | EDA — auto-picks dataset folder with most subfolders, builds `class_images`, bar chart |
| 6 | `ca45c4b5` | markdown | Sample grid header |
| 7 | `01baab98` | code | Sample grid — one image per class |
| 8 | `c49f4ecb` | markdown | HEIC conversion header |
| 9 | `ee45b750` | code | HEIC → JPG batch conversion in place, refreshes `class_images` |
| 10 | `38df18c4` | markdown | Resize header |
| 11 | `4a685ba3` | code | Resize all images to 224×224 → `processed/` as `ClassName_stem.jpg` (threaded) |
| 12 | `22863d80` | markdown | CLAHE intro |
| 13 | `b10f6a2d` | code | CLAHE contrast enhancement → `clahe/`; previews 6 before/after pairs (threaded) |
| 15 | `a57aeebc` | markdown | GrabCut intro |
| 16 | `4725a5fb` | code | GrabCut background removal → `segmented/`; previews 6 random before/after pairs (threaded) |
| 17 | `c2b16594` | markdown | Feature extraction intro |
| 18 | `d2d777c4` | code | 256-dim feature vector (192 colour + 64 Sobel gradient) from `clahe/` → `features.csv` (threaded) |
| 19 | `08b39f00` | markdown | Cosine similarity visual demo header |
| 20 | `c74c946b` | code | Visual demo — picks random image, displays top 5 most similar |
| 21 | `b860de48` | markdown | Cosine similarity validation header |
| 22 | `fcabcd34` | code | Cosine 5-NN validation — retrieval accuracy + per-class report |
| 23 | `09fc2ab6` | markdown | Train/test split header |
| 24 | `0d1b52bc` | code | 80/20 stratified split + StandardScaler |
| 25 | `bb1cd2ff` | markdown | Classifiers header |
| 26 | `c7e32b3f` | code | KNN, SVM, Random Forest, Logistic Regression, MLP — training + accuracy table |
| 27 | `6aa513f3` | markdown | Evaluation header |
| 28 | `880f4f95` | code | Summary table (all 6 methods sorted by accuracy) + confusion matrices (top 3) + best model report |
| 29 | `6bf97eb2` | markdown | Per-class F1 chart header |
| 30 | `391db60d` | code | Grouped bar chart — per-class F1 score for all 6 methods |
| 31 | `31566bd7` | markdown | Prediction demo header |
| 32 | `fe55175f` | code | Single-image prediction demo — runs all models on one random image, shows ✓/✗ per model |

## Key Variables
- `dataset_dir` — raw image folder (auto-detected; currently `date_fruit_dataset/`)
- `processed_dir` — resized image folder (`processed/`)
- `segmented_dir` — GrabCut output folder (`segmented/`)
- `class_images` — `dict[str, list[str]]` mapping class name → list of filenames (set in EDA cell, refreshed after HEIC conversion)
- `get_class(filename)` — parses class name from `ClassName_stem.jpg` by splitting on the **first** `_` (class names have no underscores)
- `clahe_dir` — CLAHE-enhanced image folder (`clahe/`)
- `features.csv` — columns: `filename`, `class`, `f0`–`f255` (192 colour + 64 Sobel gradient, both L1-normalised)
- `le` — `LabelEncoder` mapping class names to integers (set in split cell and cosine validation cell)
- `scaler` — `StandardScaler` fit on training features (set in split cell)
- `cosine_preds` — cosine 5-NN predictions on test set (set in cosine validation cell)
- `knn_preds`, `svm_preds`, `rf_preds`, `lr_preds`, `mlp_preds` — predictions on test set (set in classifiers cell)
- `knn_acc`, `svm_acc`, `rf_acc`, `lr_acc`, `mlp_acc` — float accuracies (set in classifiers cell)
- `results` — dict keyed by model name → `{clf, preds, acc}` (set in classifiers cell)
- `classifiers` — dict keyed by model name → fitted sklearn estimator (set in classifiers cell)
- `all_models` — dict keyed by model name → `(acc, preds)` including Cosine 5-NN (set in evaluation cell)

## How to Add a New Cell
1. Add a one-line markdown cell above describing what the cell does
2. Reuse `dataset_dir`, `processed_dir`, `segmented_dir`, and `class_images` — run EDA cell first
3. Import only what the cell needs
4. Use `pillow-heif` + `PIL.Image.open(...).convert("RGB")` to read any image type
5. Prefer reading from `segmented/` (background removed) → `clahe/` → `processed/` over raw `date_fruit_dataset/`

## GrabCut Notes
- Dataset background is already white (controlled lab), but background pixels dilute colour histograms — GrabCut isolates the fruit so histograms reflect only fruit colours
- Assumes fruit is centred — a `BORDER=10px` margin is treated as background (guaranteed here by the fixed imaging setup)
- Falls back to the original image if foreground mask covers < 10% of the image
- `SAMPLE_SIZE` variable controls how many random images to preview
- GrabCut boosted Cosine 5-NN accuracy from 32.53% → 72.59% by removing background pixel dilution

## Colab / Environment Notes
- Notebook kernel is set to local `ai` conda env — switch kernel to Colab when running remotely
- `drive.mount()` does **not** work when VS Code is connected to a Colab runtime (no browser UI for auth); open the notebook directly at colab.research.google.com instead
- Cell 1 auto-detects Colab via `sys.modules` and `COLAB_GPU` env var

## Cell Execution Order
Run cells in this order for a clean session:
1. pip installs (once only)
2. Colab setup
3. Extract zip → EDA → Sample grid → HEIC → Resize → CLAHE (fast) → GrabCut (long, ~10 min)
4. Feature extraction (fast, threaded — uses `clahe/` as source)
5. Cosine visual demo → Cosine 5-NN validation
6. Train/test split → KNN + SVM → Confusion matrices

The evaluation cell (`880f4f95`) requires `cosine_preds`, all `*_preds` variables, and `le` in memory — run cells 22, 24, and 26 before it.
The prediction demo (`fe55175f`) requires `scaler`, `classifiers`, `all_models`, `X_train`, `y_train`, `le`, and `normalise` in memory — run all classifier cells first.

## Latest Results (from segmented/ images, 256-dim features)
| Method | Accuracy |
|--------|----------|
| Cosine 5-NN | 72.59% |
| KNN | 79.22% |
| Random Forest | 84.94% |
| SVM | 86.14% |
| Logistic Regression | 87.35% |
| MLP | **89.16%** |

MLP per-class F1: Ajwa 1.00, Galaxy 0.92, Medjool 0.87, Meneifi 0.74, Nabtat Ali 0.94, Rutab ~0.90, Shaishe ~0.91, Sokari ~0.90, Sugaey ~0.88.
Hardest class: Meneifi (medium-brown, similar to several others).

## Planned Next Steps
- Write research paper (due June 2, 2026)
- Build PowerPoint presentation
- Consider CNN-based classification for higher accuracy on Medjool/Meneifi

---

## Course Project Requirements (ARTI 404 Guidelines)

**Institution:** Imam Abdulrahman Bin Faisal University — College of Computer Science & IT

**Due Date:** June 2nd, 2026

### Required Deliverables
1. **Research paper** (Word or PDF) — IEEE conference template preferred; source must be cited as footnote
2. **PowerPoint presentation** — presented during class evaluation
3. **Source code** — this notebook (`download_dataset.ipynb`)

### Paper Structure (Required Sections)
| Section | Content |
|---------|---------|
| I. Introduction | Purpose of the project; overview of Kaggle as data source |
| II. Goals & Objectives | Specific goals, dataset choice, image processing techniques used, performance targets |
| III. Related Works | Literature review from reputable sources; how this project differs |
| IV. Data Acquisition & Preprocessing | Dataset description, preprocessing steps — emphasize in-class techniques |
| V. Model Development & Training | Model description, training process, techniques used to improve performance |
| VI. Evaluation & Analysis | Metrics, result analysis, potential improvements |
| VII. Conclusion & Future Work | Summary of goals/results, implications, future recommendations |

### Grading Rubric (Total: 15 marks)
| Component | Marks |
|-----------|-------|
| First draft submission (Apr 19, 2026) — project idea + literature review | 2 |
| Presentation (clarity + Q&A) | 2 |
| Consistency with conference style guidelines | 1 |
| Clarity and organization of the paper | 2 |
| Quality of the research | 5 |
| Quality of the source code (documented, clear) | 3 |

### Notes
- First draft deadline (April 19, 2026) has already passed — focus on final submission
- IEEE template: https://www.ieee.org/conferences/publishing/templates.html (optional but encouraged)
- Source code quality is worth 3/15 marks — keep notebook cells well-documented with markdown headers
