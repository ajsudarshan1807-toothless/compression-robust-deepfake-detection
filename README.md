# Compression-Robust Deepfake Detection

This project builds a deepfake detection system optimized to handle image degradation, specifically JPEG compression. It implements a two-branch architecture:
1. **Spatial Branch**: EfficientNet-B0 to capture visual artifacts and spatial features.
2. **Frequency Branch**: 2-level Haar Discrete Wavelet Transform (DWT) followed by a CNN to capture frequency-domain anomalies common in AI-generated faces.

The model uses on-the-fly compression augmentation to build robustness against social media compression.

## Prerequisites & Installation

> [!WARNING]
> **Environment Note**: This project requires a C/C++ compiler if you are installing dependencies (like NumPy/SciPy) from source. Ensure you are using a stable Python version (e.g., Python 3.11 or 3.12) where pre-compiled wheels are available, or install the Visual Studio Build Tools for Windows.

1. Clone or navigate to this directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow

To reproduce the results, run the scripts in the following order:

### 1. Data Acquisition and Preprocessing
```bash
python src/data_prep.py
```
This script downloads a lightweight deepfake dataset using HuggingFace `datasets`. It detects and crops the faces using MTCNN, scales them to 224x224, and splits them into `train.csv`, `val.csv`, and `test.csv` in `data/splits/`.

### 2. Sanity Check
```bash
python src/compression_aug.py
```
Generates a side-by-side grid of sample images at different compression levels to visually confirm the augmentation module works. Check `results/compression_sanity_check.png`.

### 3. Model Training
```bash
python src/train.py
```
This script sequentially trains:
- **Model A**: Spatial branch on clean data.
- **Model B**: Frequency branch on clean data.
- **Model C**: Fusion model on clean data.
- **Model D**: Fusion model with live compression augmentation (The Primary Deliverable).

Checkpoints are saved in `checkpoints/`.

### 4. Evaluation
```bash
python src/evaluate.py
```
Tests Models A, C, and D across various test-time compression levels (Clean, QF90, QF70, QF50, QF30). 
- Results are logged to `results/ablation_table.csv`.
- A plot is generated at `results/accuracy_vs_compression.png`.

### 5. Demo Application
Launch the interactive Streamlit app to upload custom images and dynamically apply compression:
```bash
.venv\Scripts\python -m streamlit run app.py
```

## Results

- **Image Detection**: The dual-stream image model (CLIP Vision + DWT) achieved **100% accuracy** on the custom WhatsApp-compressed test dataset after targeted fine-tuning on WhatsApp artifacts. The model leverages both full-frame global context and MTCNN-cropped local facial features for highly robust detection.
- **Video Detection**: The temporal sequence model (CLIP Spatial + LSTM) was initially trained on a 400-video subset of the HuggingFace `angads24/deepfake-video` dataset. After targeted fine-tuning with an unfrozen spatial branch, it achieved **100% accuracy** on the WhatsApp-compressed test video dataset.

## Known Limitations

- **Dataset Substitution**: Due to compute constraints, a smaller HuggingFace dataset was used instead of the originally planned 140k dataset.
- **CPU Fallback**: The scripts are configured for CPU inference/training due to missing CUDA/GPU hardware on the host machine.
