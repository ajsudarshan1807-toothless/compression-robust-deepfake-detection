# AGENT INSTRUCTIONS: Build Compression-Robust Deepfake Detection System

You are an AI coding agent. Execute the steps below **in order**. Each step has a
concrete deliverable and a verification check — do not proceed to the next step
until the current step's verification passes. If a step fails, debug and retry
before moving on; do not skip ahead and leave broken components behind you.

Work inside a project directory named `deepfake-detector/`. Create this structure
first:

```
deepfake-detector/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data_prep.py
│   ├── compression_aug.py
│   ├── models/
│   │   ├── spatial_branch.py
│   │   ├── frequency_branch.py
│   │   └── fusion_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── checkpoints/
├── results/
├── app.py
├── requirements.txt
└── README.md
```

---

## STEP 0 — Environment Setup

**Action:**
1. Create `requirements.txt` with: `torch`, `torchvision`, `facenet-pytorch`,
   `pywavelets`, `opencv-python`, `pillow`, `scikit-learn`, `pandas`, `numpy`,
   `matplotlib`, `streamlit`, `tqdm`.
2. Install dependencies.
3. Verify GPU availability (`torch.cuda.is_available()`); if False, log a warning
   and proceed on CPU with reduced batch size (8 instead of 32).

**Verify:** `python -c "import torch, torchvision, pywt, cv2; print('OK')"` prints `OK`.

---

## STEP 1 — Acquire Dataset

**Action:**
1. Download the **"140k Real and Fake Faces"** dataset from Kaggle as the primary
   dataset (search Kaggle API: `kaggle datasets download -d xhlulu/140k-real-and-fake-faces`).
   Do not wait on FaceForensics++/Celeb-DF approval-gated datasets — treat those as
   optional secondary datasets only if they become available later.
2. Extract into `data/raw/`.
3. Write `src/data_prep.py` to:
   - Count real vs fake samples.
   - If class imbalance exceeds a 60/40 ratio, undersample the majority class.
   - Log final counts to `results/dataset_stats.txt`.

**Verify:** `data/raw/` contains labeled real/fake images; `dataset_stats.txt`
shows a roughly balanced class split (report the exact numbers).

---

## STEP 2 — Face Detection & Preprocessing

**Action:**
1. In `src/data_prep.py`, add a function using MTCNN (`facenet-pytorch`) to:
   - Detect the face in each image.
   - Crop with 20% margin around the bounding box.
   - Resize to 224×224.
   - Skip and log any image where no face is detected (do not crash the pipeline).
2. Save processed images to `data/processed/` preserving real/fake labels in the
   directory structure (`data/processed/real/`, `data/processed/fake/`).
3. Split into train/val/test (70/15/15). **Split by source identity/video ID if
   that metadata exists in the dataset** to prevent identity leakage across splits;
   otherwise split by filename hash to keep splits reproducible. Save split lists
   as CSVs in `data/splits/`.

**Verify:** `data/processed/` has cropped 224×224 face images; `data/splits/`
contains `train.csv`, `val.csv`, `test.csv` with no overlapping filenames between
them. Print the count of skipped (no-face-detected) images.

---

## STEP 3 — Compression Augmentation Module

**Action:**
1. Write `src/compression_aug.py` with a function `jpeg_compress(image, quality)`
   that:
   - Takes a PIL image and a quality factor (int, 1–100).
   - Saves to an in-memory buffer at that JPEG quality.
   - Reloads and returns the degraded image.
2. Write a second function `random_compress(image)` that picks a random quality
   factor from `[30, 50, 70, 90]` and applies `jpeg_compress`.
3. Generate a visual sanity-check: take 3 sample images, save a side-by-side grid
   at QF 90/70/50/30 to `results/compression_sanity_check.png`.

**Verify:** `compression_sanity_check.png` exists and shows visibly increasing
degradation from QF90 → QF30. Do not proceed until this is visually confirmed —
if degradation isn't visible, the compression function is broken.

---

## STEP 4 — Spatial Branch (Baseline Model A)

**Action:**
1. Write `src/models/spatial_branch.py`: load `EfficientNet-B0` pretrained on
   ImageNet (via `torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')`),
   replace the final classifier layer with a single output neuron.
2. Write `src/train.py` with a configurable training loop (model, dataloader,
   epochs, lr, use_compression_aug flag).
3. Train **Model A**: spatial branch only, `use_compression_aug=False`, clean
   training data only. Use Adam optimizer, lr=1e-4, batch size 32 (or 8 on CPU),
   15–20 epochs with early stopping on validation loss (patience=3).
4. Save checkpoint to `checkpoints/model_a_spatial_clean.pt`.
5. Evaluate on clean validation set; log accuracy to `results/model_a_metrics.json`.

**Verify:** Validation accuracy for Model A on clean data exceeds 90%. If it does
not, debug the training loop (check learning rate, data loading, label
correctness) before proceeding — a broken baseline invalidates every later
comparison.

---

## STEP 5 — Frequency Branch (Sanity Model B)

**Action:**
1. Write `src/models/frequency_branch.py`:
   - Function `dwt_features(image)`: apply 2-level Haar DWT via `pywt.dwt2`,
     extract LH/HL/HH subbands, stack as input channels.
   - A small CNN: 4 conv layers (increasing channels 16→32→64→128) + BatchNorm +
     ReLU + MaxPool, followed by global average pooling → 256-d output vector.
2. Attach a single-neuron classification head directly to this branch (temporary,
   for standalone testing only — this head is discarded after this step).
3. Train **Model B**: frequency branch alone, clean training data, same
   hyperparameters as Step 4.
4. Log accuracy to `results/model_b_metrics.json`.

**Verify:** Model B accuracy is meaningfully above 50% (random chance) on clean
validation data — e.g. >65%. It does not need to beat Model A; it only needs to
confirm the frequency branch has learned a real signal, not noise. If it's at or
near chance level, debug the DWT feature extraction before proceeding to fusion.

---

## STEP 6 — Fusion Model

**Action:**
1. Write `src/models/fusion_model.py`:
   - Load spatial branch (EfficientNet-B0, feature output before final layer,
     1280-d) and frequency branch (from Step 5, 256-d, discard its temporary head).
   - Concatenate → 1536-d vector.
   - FC(1536→512) + ReLU + Dropout(0.5) → FC(512→1) → Sigmoid.
2. Train **Model C**: fusion architecture, `use_compression_aug=False`, clean
   training data only.
3. Train **Model D**: fusion architecture, `use_compression_aug=True` — this
   applies `random_compress()` from Step 3 to each training sample on the fly,
   per batch, per epoch (not pre-baked as a static augmented dataset).
4. Save checkpoints: `checkpoints/model_c_fusion_clean.pt`,
   `checkpoints/model_d_fusion_compaug.pt`. **Model D is the primary deliverable
   of this project.**

**Verify:** Both Model C and Model D reach validation accuracy comparable to or
better than Model A on clean data (fusion should not underperform the spatial-only
baseline on clean data — if it does, check that gradients are flowing to both
branches, not just one).

---

## STEP 7 — Compression-Robustness Evaluation

**Action:**
1. Write `src/evaluate.py`:
   - Load the test split.
   - For each compression level in `[clean, QF90, QF70, QF50, QF30]`, re-compress
     the entire test set at that level (test-time only — never train on
     compressed-and-fixed test data).
   - Run Models A, C, and D on each compressed test set version.
   - Compute Accuracy, Precision, Recall, F1, AUC-ROC (`sklearn.metrics`) for
     every (model, compression level) pair.
2. Save results as a table to `results/ablation_table.csv` with rows = models,
   columns = compression levels × metrics.
3. Generate `results/accuracy_vs_compression.png`: a line chart, x-axis =
   compression level (clean→QF30), y-axis = accuracy, one line per model
   (A, C, D). This is the single most important output of the project.

**Verify:** `ablation_table.csv` and `accuracy_vs_compression.png` exist and are
non-empty. Explicitly check and report: does Model D show a **smaller accuracy
drop-off** than Model A as compression increases? State the actual numbers — do
not claim success without the data supporting it. If Model D does not
outperform Model A under compression, report this honestly and note it as a
limitation/discussion point rather than altering the narrative.

---

## STEP 8 — Demo Application

**Action:**
1. Write `app.py` using Streamlit:
   - File uploader for an image.
   - Run Model D on the uploaded image, display Real/Fake prediction + confidence.
   - Add a slider (quality factor 10–100) that re-compresses the uploaded image
     live and re-runs prediction, showing how confidence changes with compression.
2. Test locally: `streamlit run app.py`.

**Verify:** App launches without errors; uploading a test image produces a
prediction; moving the compression slider visibly changes the displayed
confidence score.

---

## STEP 9 — Final Report Artifacts

**Action:**
1. Write `README.md` summarizing: problem statement, architecture, dataset used,
   how to reproduce (setup → train → evaluate → demo commands), and final results
   table.
2. Copy `results/ablation_table.csv` and `results/accuracy_vs_compression.png`
   into the README as embedded results.
3. List any deviations from the original plan (e.g. dataset substitutions, any
   step whose verification did not fully pass) explicitly under a "Known
   Limitations" section — do not omit them.

**Verify:** `README.md` is self-contained enough that a new team member could
reproduce every result by following it top to bottom.

---

## Rules for the Whole Run

- Never fabricate or hand-write metric numbers — all numbers in any output file
  must come from actual executed code and real evaluation runs.
- Do not skip the clean-data baseline (Model A) or the standalone frequency
  sanity check (Model B) even though they are not the final deliverable — every
  later comparison depends on them existing and being verified.
- Do not apply compression augmentation to test data before evaluation time — it
  must only be a training-time augmentation, applied fresh at test time per
  compression level in Step 7.
- Split data by identity/source, not by raw filename shuffling, wherever
  identity metadata is available — this project's validity is undermined by
  train/test leakage.
- If any verification step fails after reasonable debugging attempts, stop, and
  clearly report which step failed and why, rather than proceeding on top of a
  broken component.
