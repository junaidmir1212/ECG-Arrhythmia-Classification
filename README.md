# 🫀 Clinically Explainable ECG Arrhythmia Diagnostic Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecg-arrhythmia-diagnostics.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Clinical Standard](https://img.shields.io/badge/AAMI-EC57%20Compliant-success.svg)](https://www.aami.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical decision-support system (CDSS) for automated ECG rhythm analysis, beat-level arrhythmia classification, and continuous triage. Built with PyTorch, Streamlit, and Plotly, this platform implements the **AAMI EC57 inter-patient evaluation standard**, real-time **Pan-Tompkins QRS detection**, **Heart Rate Variability (HRV)** profiling, **ST-segment ischemic deviation measurement**, and **1D Grad-CAM interpretability**, complete with automated **PDF clinical audit report generation**.

---

## 🌟 Key Capabilities

- **AAMI EC57 Inter-Patient Cohort Evaluation**: Trained and validated using rigorous patient-decoupled splits ($DS1$ for training/validation, $DS2$ for testing) across 5 standard diagnostic superclasses:
  - **N**: Normal Sinus / Bundle Branch Blocks
  - **S**: Supraventricular Ectopic Beats (Atrial Premature / Aberrant)
  - **V**: Ventricular Ectopic Beats (Premature Ventricular Contractions / PVCs)
  - **F**: Fusion Beats (Normal + Ventricular Conduction)
  - **Q**: Paced / Unclassifiable Beats
- **Autonomous Continuous Rhythm Scanning**: Processes raw continuous Lead II telemetry traces (`.csv`, `.txt`, `.npy`) via an integrated **Pan-Tompkins QRS detection engine** (bandpass filtering, differentiation, squaring, moving-window integration, and local peak refinement).
- **Interactive Click-to-Scrub Rhythm Strip**: Full Plotly continuous waveform with color-coded ectopic beat triage markers. Click any peak along the recording to immediately isolate and inspect its morphology.
- **Autonomic Tone & Heart Rate Variability (HRV)**: Automated calculation of time-domain autonomic biomarkers:
  - **Mean R-R** & instantaneous **Mean Heart Rate (bpm)**
  - **SDNN** (Global autonomic variability)
  - **RMSSD** (Parasympathetic/vagal regulatory tone)
  - **pNN50** & Interactive **R-R Tachogram** visualization
- **Myocardial Ischemia & STEMI Triage**: Algorithmic measurement of $J$-point and $J+60\text{ ms}$ deflection relative to the isoelectric $PR$ segment baseline. Automatically flags:
  - 🚨 **STEMI Alert**: Acute ST-elevation ($\ge +0.10\text{ mV}$)
  - ⚠️ **Ischemia Alert**: Significant ST-depression ($\le -0.05\text{ mV}$)
  - ✅ **Normal Repolarization**
- **1D Grad-CAM Diagnostic Waveform Attribution**: Saliency heatmaps overlaid on the extracted ECG waveforms highlight exact temporal windows (e.g., widened QRS complex, compensatory pauses, or hyperacute repolarization) that drive neural predictions.
- **Automated Clinical Audit PDF Export**: Single-click compilation into an audit-ready medical report via `fpdf2`, documenting patient trace metrics, AAMI class distributions, HRV tables, ST measurements, and high-resolution Grad-CAM diagnostic figures.

---

## 📐 Architecture & Mathematical Foundations

### 1. Neural Classifier Architecture
The backbone model (`ECGArrhythmiaClassifier`) employs a multi-stage 1D convolutional residual network optimized for electrophysiological feature extraction:

```text
Input ECG Beat (1 × 256 @ 360 Hz)
  │
  ├── Conv1D (k=7, s=1, p=3) ──► BatchNorm ──► LeakyReLU ──► MaxPool (s=2)
  ├── Conv1D (k=5, s=1, p=2) ──► BatchNorm ──► LeakyReLU ──► MaxPool (s=2)
  ├── Conv1D (k=3, s=1, p=1) ──► BatchNorm ──► LeakyReLU ──► [Target Layer for Grad-CAM]
  │
  ├── AdaptiveAvgPool1D(1)
  ├── Flatten
  ├── Dropout (p=0.4)
  └── Linear (Hidden ──► 5 Logits) ──► Softmax Output