# How the Code Works — Plain English Explanation
### ARTI 404 Date Fruit Classification Project

---

## The Big Picture

We have 1,658 photos of date fruits across 9 varieties:

| Class | Images |
|-------|--------|
| Ajwa | 175 |
| Galaxy | 190 |
| Medjool | 135 |
| Meneifi | 232 |
| Nabtat Ali | 177 |
| Rutab | 146 |
| Shaishe | 171 |
| Sokari | 264 |
| Sugaey | 168 |

The goal is to build a system that can look at a photo and correctly identify which variety of date it is.

The pipeline has 5 main stages:
1. **Prepare the images** — resize, enhance contrast, remove background
2. **Describe each image as numbers** — turn a photo into a list of 256 numbers
3. **Train classifiers** — show them labelled examples so they learn the patterns
4. **Test and evaluate** — check how accurate they are on photos they've never seen
5. **Compare methods** — see which approach works best

**Final results:**
| Method | Accuracy |
|--------|----------|
| Cosine 5-NN | 72.59% |
| KNN | 79.22% |
| Random Forest | 84.94% |
| SVM | 86.14% |
| Logistic Regression | 87.35% |
| MLP | **89.16%** |

---

## Stage 1 — Preparing the Images

### Step 1: Resize to 224×224
**Output folder:** `processed/`

Every photo in the dataset is a different size. A phone photo might be 4000×3000 pixels, another might be 800×600. Before we can compare images or train a model, they all need to be the same size.

We resize every image to **224×224 pixels**. This is a standard size used in computer vision — small enough to process quickly, large enough to still see the fruit clearly.

The file also gets renamed to `ClassName_originalname.jpg` (e.g. `Ajwa_IMG_0042.jpg`) so we always know which class an image belongs to just from its filename.

We use 8 parallel workers (threads) to do this fast — all 1,658 images get processed simultaneously instead of one at a time.

---

### Step 2: CLAHE — Contrast Enhancement
**Output folder:** `clahe/`

Some photos are taken in dim lighting, others in bright sunlight. A dark photo of an Ajwa date and a bright photo of the same Ajwa date look very different to a computer, even though they're the same fruit.

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) fixes this. Here's what it does step by step:

1. Converts the image from RGB colour to **LAB colour space**. LAB separates "how bright" (L channel) from "what colour" (A and B channels). This is important because we only want to adjust brightness, not colour.
2. Applies histogram equalization only to the **L (lightness) channel** — it boosts contrast in dark areas and tones down overexposed areas, without changing any colours.
3. Converts back to the original colour format and saves.

Think of it like the "auto-enhance" button on your phone's photo editor — it makes the fruit detail clearer without distorting the colours.

The `clipLimit=2.0` setting prevents over-enhancement (which would create unnatural-looking images). The `tileGridSize=(8,8)` means it applies the enhancement in small 8×8 pixel tiles, so it works locally — brightening a dark corner without washing out an already-bright area.

---

### Step 3: GrabCut — Background Removal
**Output folder:** `segmented/`

This is the **most impactful** preprocessing step.

**About the dataset:** These images were taken in a carefully controlled lab environment (from the paper *"Classification of Date Fruits in a Controlled Environment Using CNNs"* by Alhamdan & Howe, 2021). The setup used a Canon EOS 550D DSLR with flash, a 48cm ring light at 100% brightness to eliminate shadows, a white paper background, and a fixed camera-to-fruit distance of 8cm. Every single photo was shot against a **white background** under consistent lighting.

So unlike a "wild" dataset, the background here is already white and consistent. Why does GrabCut still help then?

**The problem with white background pixels in feature extraction:**

Each image is 224×224 = 50,176 pixels. The actual date fruit only occupies the centre — maybe 20,000–30,000 of those pixels. The rest are white background. When we compute a colour histogram over the whole image, the white pixels (which all land in the brightest bins of R, G, and B) heavily dominate the histogram. A small dark Ajwa fruit surrounded by lots of white pixels produces a histogram that looks more "white" than "dark" — the fruit's actual colour signature is diluted by the background.

GrabCut fixes this by painting the background a uniform white and allowing us to clearly isolate the fruit region. Once segmented, the colour histograms reflect the **fruit's true colours** rather than a mix of fruit + background.

**How GrabCut works:**

1. It assumes the fruit is roughly centred in the image — which is guaranteed here since all photos were taken with the same controlled setup.
2. It marks a 10-pixel border around all four edges as **"definitely background."** (These are always white background pixels in this dataset.)
3. Everything inside that border is marked as **"probably foreground."**
4. It runs an iterative algorithm (5 iterations) that looks at colour differences to find the precise boundary between the fruit edge and the background.
5. It applies morphological operations (dilation then erosion using an elliptical kernel) to smooth the boundary mask and fill any small holes.
6. Background pixels are replaced with **pure white (255, 255, 255)** — making the boundary perfectly clean.
7. Safety check: if less than 10% of the image is detected as foreground (meaning GrabCut failed to find the fruit), the original image is kept instead.

All 1,658 images complete in about **1 minute 55 seconds** using 4 parallel workers.

**Why this still made a huge difference — proven by the numbers:**

Even though the background was already white, using GrabCut-segmented images caused the Cosine 5-NN accuracy to jump from **32.53% → 72.59%**. The reason: before segmentation, white background pixels were diluting the colour histograms of every image similarly, making all 9 classes look more alike than they really are. After segmentation, each image's histogram captures only the fruit, making the colour fingerprint much more distinctive per class.

Nabtat Ali F1 jumped from **0.73 → 0.96**. Sugaey went from **0.67 → 0.84**. These classes likely had fruit that was smaller relative to frame size, so their colour signal was most diluted by background pixels before segmentation.

---

## Stage 2 — Turning Images into Numbers (Feature Extraction)
**Output:** `features.csv`
**Speed:** 920 images/second (completes in ~2 seconds)

A computer cannot "look" at a photo the way you do. It only understands numbers. So the first question is: **how do you turn a photo into a useful list of numbers?**

A 224×224 image is technically already a grid of numbers — each pixel has three values (R, G, B), giving 224 × 224 × 3 = **150,528 numbers** per image. But that raw form is terrible for classification:
- Two photos of the same fruit taken 1cm apart would produce completely different 150,528-number lists
- The raw numbers encode position — pixel 5,000 in one image means something completely different in another
- It's way too many numbers to compare efficiently

So instead we **summarise** the image — we throw away position and ask two simpler questions: *what colours are present?* and *how much texture/edge detail is there?* The answers to those two questions, combined, give us **256 numbers** that are compact, comparable, and meaningful.

---

### Feature Group 1: Colour Histogram (192 numbers)

**The question it answers:** *"What proportion of this image is each shade of colour?"*

**Step by step:**

1. Take the image and look at just the Red channel — every pixel has a Red value from 0 (pure black) to 255 (maximum red).
2. Divide that range (0–255) into **64 equal buckets**. Bucket 0 = very dark red (values 0–3), Bucket 63 = maximum brightness (values 252–255).
3. Go through every pixel and drop it into whichever bucket matches its Red value. Count how many pixels land in each bucket.
4. Divide every bucket count by the total number of pixels, so each bucket becomes a percentage. This is called **normalisation**.
5. Repeat steps 1–4 for the Green channel, then the Blue channel.

Result: 64 (Red) + 64 (Green) + 64 (Blue) = **192 numbers** describing the colour makeup of the image.

**Why this is useful:** Ajwa dates are almost black-purple — their histogram has high values in the very dark buckets of all three channels, and near-zero in the bright buckets. Sugaey dates are golden-yellow — their histogram peaks in medium-to-high Red, medium Green, and low Blue. These "shapes" are different enough that a classifier can tell the varieties apart.

**The white background:** Since we're using segmented images, all background pixels are pure white (R=255, G=255, B=255) — they all land in bucket 63 of every channel. Because every image has the same white background, bucket 63 is consistently high for all classes and carries no useful information for distinguishing them. The classifier effectively ignores it and pays attention to the fruit-coloured buckets instead.

---

### Feature Group 2: Gradient Magnitude Histogram (64 numbers)

**The question it answers:** *"How much surface texture and edge sharpness does this fruit have?"*

Colour alone isn't enough — some date varieties share similar brown tones but have very different skin textures. This is where **Sobel edge detection** comes in.

**How Sobel works:**

The Sobel operator looks at each pixel and measures how much the brightness changes compared to its neighbours:
- In a smooth, flat area (like the glossy side of a Rutab date) — neighbouring pixels have similar brightness → **small change → low gradient value**
- At a wrinkle, ridge, or texture pattern (like the rough skin of a Medjool) — neighbouring pixels have very different brightness → **large change → high gradient value**

It does this in two passes:
- One pass detects horizontal changes (left-right edges)
- One pass detects vertical changes (up-down edges)

Then combines them: `gradient = sqrt(horizontal² + vertical²)`

Every pixel now has a single "how much edge is here" value instead of a colour.

We then do the same bucketing process as before — divide the gradient range into **64 buckets** and count how many pixels fall in each, then normalise. Result: **64 numbers** describing the texture profile of the fruit.

**Why this helps:** Medjool dates are famously wrinkled and fibrous — their gradient histogram skews toward the high-edge buckets. Rutab dates in their wet/glossy stage are much smoother — their histogram skews toward low-edge buckets. When two varieties have similar colour (like Meneifi and Medjool, both medium-brown), the texture difference helps the classifier make the right call.

---

### The Complete Feature Vector

Combining both groups gives **256 numbers per image** — a compact fingerprint capturing both colour (192) and texture (64).

Think of it like a wine sommelier's tasting notes — instead of looking at the full 750ml bottle, they summarise it as: "dark ruby, dry, high tannins, notes of cherry." That summary is enough to identify the wine. Our 256 numbers do the same job for date fruit photos.

All 1,658 fingerprints are saved to `features.csv`:
- Each row = one image
- Column 1 = filename (e.g. `Ajwa_IMG_0042.jpg`)
- Column 2 = class name (e.g. `Ajwa`)
- Columns 3–258 = the 256 feature values (`f0` through `f255`)

---

## Stage 3 — Training the Classifiers

### Train/Test Split

Before training, we split the 1,658 images:
- **Training set (80% = 1,326 images):** the classifier sees these and learns
- **Test set (20% = 332 images):** never seen during training — used only for the final accuracy check

We use **stratified** splitting — each class is split 80/20 proportionally. So Medjool (135 images) contributes ~108 to training and ~27 to testing. No class is accidentally left out of testing.

We also apply **StandardScaler** — rescales all 256 features so each has a mean of 0 and standard deviation of 1. Without this, features with naturally large values would dominate the distance calculations unfairly.

---

### Classifier 1: Cosine Similarity 5-NN — **72.59% accuracy**

The simplest approach — no real "training" happens. Pure similarity search.

**How it works:**
1. For each test image, compute its cosine similarity against all 1,326 training images.
2. Cosine similarity measures how aligned two 256-number vectors are. Two identical vectors score 1.0; completely opposite vectors score -1.0.
3. Find the 5 most similar training images (highest cosine scores).
4. Majority vote among those 5 — whichever class appears most is the prediction.

**Accuracy: 72.59%** — good baseline, no learning needed. Best class: Ajwa (F1=0.99). Hardest: Meneifi (F1=0.57).

---

### Classifier 2: KNN (K-Nearest Neighbours) — **79.22% accuracy**

Very similar to cosine 5-NN but uses scikit-learn's optimised implementation with StandardScaler applied.

**How it works:**
1. During "training," KNN memorises all 1,326 training examples — no actual learning.
2. For each test image, finds the 5 nearest training images in 256-dimensional feature space.
3. Majority vote gives the prediction.

**Accuracy: 79.22%** — better than raw cosine because StandardScaler ensures all 256 features contribute equally to the distance calculation.

---

### Classifier 3: Random Forest

A Random Forest trains **200 decision trees**, each on a random subset of the training data and features. Each tree independently votes for a class, and the majority vote wins.

Individual trees overfit easily — but 200 of them, each seeing slightly different data, tend to cancel out each other's mistakes. This is called **ensemble learning**.

---

### Classifier 4: Logistic Regression

Despite the name, this is a classification algorithm. It learns a linear boundary between classes in feature space, but uses a sigmoid/softmax function to output a probability for each class.

It's fast, interpretable, and acts as a strong linear baseline. If the classes are linearly separable in 256-dimensional space, Logistic Regression handles it cleanly.

---

### Classifier 5: MLP (Multi-Layer Perceptron Neural Network)

A small neural network with two hidden layers (256 neurons → 128 neurons). It learns non-linear combinations of the 256 input features through matrix multiplications and ReLU activations.

This is the simplest form of deep learning. With only 1,326 training samples it won't outperform SVM dramatically, but it adds a neural network comparison to the paper.

---

### Classifier 6: SVM (Support Vector Machine) — **86.14%**

The most sophisticated classifier. It actually learns from the data.

**How it works (simplified):**

Imagine each of the 1,326 training images as a point in 256-dimensional space. Each class forms a cloud of points. SVM finds the best **boundaries** (called hyperplanes) that separate each class from the others, with the **maximum possible gap** (margin) between the boundary and the nearest points of each class.

We use the **RBF (Radial Basis Function) kernel**, which means the boundaries can be curved. This is important because real-world fruit data doesn't separate neatly in straight lines.

The `C=10` parameter controls strictness — higher C means the SVM tries hard to classify every training point correctly, accepting a narrower margin. `gamma="scale"` automatically scales the kernel width based on the number of features.

**Accuracy: 86.14%** — finds optimal decision boundaries rather than just voting. Strong, but beaten by Logistic Regression and MLP on this dataset.

---

## Stage 4 — Evaluation Results

### Final Accuracy Comparison
| Method | Accuracy | Training time |
|--------|----------|--------------|
| Cosine 5-NN | 72.59% | — (no training) |
| KNN | 79.22% | ~0.0s |
| Random Forest | 84.94% | ~0.3s |
| SVM | 86.14% | ~0.1s |
| Logistic Regression | 87.35% | ~0.1s |
| MLP | **89.16%** | ~0.8s |

All models train in under 1 second — the dataset is small enough that even the neural network is fast.

### Per-Class MLP Results (Best Model)

| Class | Precision | Recall | F1 | What it means |
|-------|-----------|--------|----|---------------|
| Ajwa | 1.00 | 1.00 | **1.00** | Perfect — Ajwa's near-black colour is completely unique |
| Galaxy | 0.90 | 0.95 | **0.92** | Very good |
| Medjool | 0.88 | 0.85 | **0.87** | Good — improved significantly from SVM |
| Nabtat Ali | 0.97 | 0.91 | **0.94** | Very good — distinctive colour profile |
| Rutab | ~0.84 | ~0.93 | **~0.90** | Very good — glossy wet-stage appearance |
| Shaishe | ~0.94 | ~0.88 | **~0.91** | Very good |
| Sokari | ~0.89 | ~0.91 | **~0.90** | Very good |
| Sugaey | ~0.85 | ~0.88 | **~0.88** | Good |
| Meneifi | 0.80 | 0.68 | **0.74** | Hardest — medium brown, similar colour to several others |

Meneifi is the only class below 0.8 F1. Its medium-brown colour profile overlaps with Medjool, Sokari, and Sugaey. A CNN that captures spatial structure would help here.

### What Precision and Recall Mean

- **Precision:** "Of all the images I labelled as Ajwa, what fraction were actually Ajwa?" — measures false alarms. Ajwa precision = 1.00 means every single image the model called Ajwa was genuinely Ajwa.
- **Recall:** "Of all the actual Ajwa images, how many did I correctly find?" — measures missed detections. Ajwa recall = 0.91 means 9% of real Ajwa images were wrongly labelled as something else.
- **F1-score:** The average of precision and recall — one balanced score. 1.0 is perfect.

---

## Stage 5 — Why Each Step Improved Results

| Step | Effect on Accuracy | Why |
|------|--------------------|-----|
| Resize to 224×224 | Enables comparison | All images must be the same size |
| CLAHE | Removes lighting variation | A dark and a bright photo of the same fruit produce more similar histograms |
| GrabCut | **+40 points on Cosine, +5 on KNN/SVM** | Removes background noise entirely — features now describe only the fruit |
| Gradient (Sobel) features | Improves texture discrimination | Helps separate varieties with similar colour but different skin texture |
| SVM over KNN | +7 points | Learns optimal decision boundaries instead of just voting |

---

## The Folder Structure

```
project/
├── date_fruit_dataset/     <- original photos, one subfolder per class
│   ├── Ajwa/               <- 175 images
│   ├── Galaxy/             <- 190 images
│   └── ...
├── processed/              <- all images resized to 224x224 (1,658 files)
├── clahe/                  <- contrast-enhanced versions of processed/ (1,658 files)
├── segmented/              <- GrabCut background-removed (1,658 files)
├── features.csv            <- 256 numbers per image (1,658 rows x 258 columns)
└── download_dataset.ipynb  <- the notebook with all the code
```

---

## Summary

We went from raw photos → clean, background-removed images → 256-number fingerprints → a classifier that correctly identifies the date variety **89.16%** of the time across 9 classes, with Ajwa reaching a perfect 1.00 F1.

The single most impactful step was **GrabCut background removal** — it alone caused cosine similarity accuracy to jump from 32% to 72%, proving background noise was the dominant source of confusion. The MLP neural network then learned non-linear combinations of colour and texture features to push accuracy to 89%.

The only remaining hard class is Meneifi (F1=0.74) — its medium-brown colour overlaps with several other varieties. Improving on it would require a CNN that captures spatial structure, or more training images.
