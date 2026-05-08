# Dates Fruit Image Classification — ARTI 404

## Project Overview
This project classifies 9 types of date fruits using image processing and computer vision.
Current dataset: 1,658 images across 9 classes sourced from Kaggle (`archive.zip`).

## Dataset Structure
- **Raw images:** `date_fruit_dataset/` — one subfolder per class
- **Classes (9):** Ajwa (175), Galaxy (190), Medjool (135), Meneifi (232), Nabtat Ali (177), Rutab (146), Shaishe (171), Sokari (264), Sugaey (168)
- **Processed images:** `processed/` — all images resized to 224×224 JPEG, named `ClassName_stem.jpg`
- **Segmented images:** `segmented/` — GrabCut background-removed versions of `processed/`
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
| 12 | `a57aeebc` | markdown | GrabCut intro |
| 13 | `4725a5fb` | code | GrabCut background removal → `segmented/`; previews 6 random before/after pairs (threaded) |
| 14 | `c2b16594` | markdown | Feature extraction intro |
| 15 | `d2d777c4` | code | 192-dim RGB colour histogram per segmented image → `features.csv` (threaded) |
| 16 | `08b39f00` | markdown | Cosine similarity visual demo header |
| 17 | `c74c946b` | code | Visual demo — picks random image, displays top 5 most similar |
| 18 | `b860de48` | markdown | Cosine similarity validation header |
| 19 | `fcabcd34` | code | Cosine 5-NN validation — retrieval accuracy + per-class report |
| 20 | `09fc2ab6` | markdown | Train/test split header |
| 21 | `0d1b52bc` | code | 80/20 stratified split + StandardScaler |
| 22 | `bb1cd2ff` | markdown | Classifiers header |
| 23 | `c7e32b3f` | code | KNN (k=5, cosine) + SVM (RBF, C=10) training and accuracy |
| 24 | `6aa513f3` | markdown | Evaluation header |
| 25 | `880f4f95` | code | 3-way confusion matrices (Cosine 5-NN / KNN / SVM) + summary table + best model report |

## Key Variables
- `dataset_dir` — raw image folder (auto-detected; currently `date_fruit_dataset/`)
- `processed_dir` — resized image folder (`processed/`)
- `segmented_dir` — GrabCut output folder (`segmented/`)
- `class_images` — `dict[str, list[str]]` mapping class name → list of filenames (set in EDA cell, refreshed after HEIC conversion)
- `get_class(filename)` — parses class name from `ClassName_stem.jpg` by splitting on the **first** `_` (class names have no underscores)
- `features.csv` — columns: `filename`, `class`, `f0`–`f191` (normalised RGB histogram, 64 bins per channel)
- `le` — `LabelEncoder` mapping class names to integers (set in split cell and cosine validation cell)
- `scaler` — `StandardScaler` fit on training features (set in split cell)
- `cosine_preds` — cosine 5-NN predictions on test set (set in cosine validation cell)
- `knn_preds`, `svm_preds` — KNN/SVM predictions on test set (set in classifiers cell)
- `knn_acc`, `svm_acc` — float accuracies (set in classifiers cell)

## How to Add a New Cell
1. Add a one-line markdown cell above describing what the cell does
2. Reuse `dataset_dir`, `processed_dir`, `segmented_dir`, and `class_images` — run EDA cell first
3. Import only what the cell needs
4. Use `pillow-heif` + `PIL.Image.open(...).convert("RGB")` to read any image type
5. Prefer reading from `processed/` (uniform 224×224 JPEGs) or `segmented/` over raw `date_fruit_dataset/`

## GrabCut Notes
- Assumes fruit is centred — a `BORDER=10px` margin around the image is treated as background
- Falls back to the original image if foreground mask covers < 10% of the image
- Images where the fruit fills the whole frame (dates in a box) are hard cases for GrabCut
- `SAMPLE_SIZE` variable controls how many random images to preview

## Colab / Environment Notes
- Notebook kernel is set to local `ai` conda env — switch kernel to Colab when running remotely
- `drive.mount()` does **not** work when VS Code is connected to a Colab runtime (no browser UI for auth); open the notebook directly at colab.research.google.com instead
- Cell 1 auto-detects Colab via `sys.modules` and `COLAB_GPU` env var

## Cell Execution Order
Run cells in this order for a clean session:
1. pip installs (once only)
2. Colab setup
3. Extract zip → EDA → Sample grid → HEIC → Resize → GrabCut (long, ~10 min)
4. Feature extraction (fast, threaded)
5. Cosine visual demo → Cosine 5-NN validation
6. Train/test split → KNN + SVM → Confusion matrices

The evaluation cell (`880f4f95`) requires `cosine_preds`, `knn_preds`, `svm_preds`, and `le` all in memory — run cells 19, 21, and 23 before it.

## Planned Next Steps
- Write research paper (due June 2, 2026)
- Build PowerPoint presentation
- Consider CNN-based classification for higher accuracy

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
