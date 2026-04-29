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
| Cell | Purpose |
|------|---------|
| pip installs | Commented out — uncomment and run once (`pillow-heif` required) |
| Extract zip | Unzips `4639543.zip` into `date_fruit_dataset/` |
| EDA — class distribution | Builds `class_images` dict, prints counts, bar chart |
| Sample grid | Shows one image per class |
| HEIC → JPG converter | Batch converts all `.heic` to `.jpg`, refreshes `class_images` |
| Resize | Resizes all images to 224×224, saves to `processed/` |

## Key Variables
- `dataset_dir` — raw image folder (`date_fruit_dataset/`)
- `processed_dir` — resized image folder (`processed/`)
- `class_images` — `dict[str, list[str]]` mapping class name → list of filenames (set in EDA cell, refreshed after HEIC conversion)
- `get_class(filename)` — parses class name from a filename

## How to Add a New Cell
1. Add a one-line markdown cell above describing what the cell does
2. Reuse `dataset_dir`, `processed_dir`, and `class_images` — run EDA cell first
3. Import only what the cell needs
4. Use `pillow-heif` + `PIL.Image.open(...).convert("RGB")` to read any image type
5. Prefer reading from `processed/` (uniform 224×224 JPEGs) over raw `date_fruit_dataset/`

## Planned Next Steps
- Color histogram feature extraction (RGB + HSV) → `features.csv`
- Similarity search using cosine distance on histograms
- Background removal / date segmentation
- Train/val/test split when full dataset arrives
