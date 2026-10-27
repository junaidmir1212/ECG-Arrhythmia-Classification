# ECG Arrhythmia Classification with Explainable AI

An end-to-end deep learning pipeline for automated cardiac arrhythmia detection using the MIT-BIH Arrhythmia Database. Evaluated under the strict **AAMI EC57 standard** with an inter-patient train/test split to prevent patient data leakage.

## Architecture
- **Preprocessing:** 0.5–45 Hz Bandpass Butterworth filtering + R-peak beat segmentation (256 samples).
- **Model:** 3-stage 1D-CNN (morphology extractor) + Bidirectional LSTM (rhythm modeling).
- **Optimization:** Multiclass Focal Loss with inverse class weighting.
- **Explainability:** 1D Grad-CAM diagnostic attribution heatmaps.

## How to Run
```bash
python download_data.py
python preprocess.py
python train.py
python evaluate.py
python explain.py