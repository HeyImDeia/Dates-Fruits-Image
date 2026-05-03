# Dates Fruit Image Classification — ARTI 404

## Project Overview
This project classifies 27 types of date fruits using image processing and computer vision.
Current dataset: ~478 images (small subset). Full dataset (~3,228 images, 27 classes) is pending arrival.
Source: [Zenodo record 4639543](https://zenodo.org/records/4639543).

## Dataset Structure
- **Raw images:** `date_fruit_dataset/` — flat folder, no subfolders per class
- **Processed images:** `processed/` — all images resized to 224×224 JPEG
- **Class parsing:** class name is the filename prefix before the trailing `-number` or `_number` (e.g. `Muraaya-1.jpg` → `Muraaya`, `SequeeIRAQ_6163.JPG` → `SequeeIRAQ`)
- **File types:** originally `.jpg`, `.jpeg`, `.png`, `.heic` — after step 1 all `.heic` are converted to `.jpg`
- **HEIC support:** requires `pillow-heif` — call `register_heif_opener()` before opening images

## Notebook Layout (`download_dataset.ipynb`)
| Extract zip | Unzips `4639543.zip` into `date_fruit_dataset/` |
| # | Cell | Purpose |
|---|------|---------|
| 0 | pip installs | Commented out — uncomment and run once (`pillow-heif`, `opencv-python` required) |
| 1 | Colab setup | Auto-detects Colab, mounts Drive, `chdir` to project folder — skipped locally |
| 2 | Extract zip | Unzips `4639543.zip` into `date_fruit_dataset/` |
| 3 | EDA — class distribution | Builds `class_images` dict, prints counts, bar chart |
| 4 | Sample grid | Shows one image per class |
| 5 | HEIC → JPG converter | Batch converts all `.heic` to `.jpg`, refreshes `class_images` |
| 6 | Resize | Resizes all images to 224×224, saves to `processed/` |
| 7 | GrabCut segmentation | Removes background from a random sample of 6 images, plots before/after |
| 8 | Feature extraction | Computes 192-dim colour histogram (64 bins × RGB) per image, saves `features.csv` |
| 9 | Cosine similarity search | Picks a random image, finds top 5 most similar images, displays as a grid |

## Key Variables
- `dataset_dir` — raw image folder (`date_fruit_dataset/`)
- `processed_dir` — resized image folder (`processed/`)
- `segmented_dir` — GrabCut output folder (`segmented/`)
- `class_images` — `dict[str, list[str]]` mapping class name → list of filenames (set in EDA cell, refreshed after HEIC conversion)
- `get_class(filename)` — parses class name from a filename
- `features.csv` — columns: `filename`, `class`, `f0`–`f191` (normalised RGB histogram, 64 bins per channel)

## How to Add a New Cell
1. Add a one-line markdown cell above describing what the cell does
2. Reuse `dataset_dir`, `processed_dir`, and `class_images` — run EDA cell first
3. Import only what the cell needs
4. Use `pillow-heif` + `PIL.Image.open(...).convert("RGB")` to read any image type
5. Prefer reading from `processed/` (uniform 224×224 JPEGs) over raw `date_fruit_dataset/`

## GrabCut Notes
- Assumes fruit is centred — a `BORDER=10px` margin around the image is treated as background
- Falls back to the original image (labelled red) if foreground mask covers < 10% of the image
- Images where the fruit fills the whole frame (e.g. dates in a box) are hard cases for GrabCut
- `SAMPLE_SIZE` variable controls how many random images to test

## Colab / Environment Notes
- Notebook kernel is set to local `ai` conda env — switch kernel to Colab when running remotely
- `drive.mount()` does **not** work when VS Code is connected to a Colab runtime (no browser UI for auth); open the notebook directly at colab.research.google.com instead
- Cell 1 auto-detects Colab via `sys.modules` and `COLAB_GPU` env var

## Planned Next Steps
- Train/val/test split when full dataset arrives
- Classification model (CNN or feature-based SVM/KNN on `features.csv`)
- Evaluate GrabCut quality more rigorously; consider SAM for hard cases
