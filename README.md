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

  ### 2. 1D Gradient-Weighted Class Activation Mapping (Grad-CAM)
Class-discriminative localization maps $L_{\text{Grad-CAM}}^c$ are computed by taking the gradient of class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{L} \sum_{i=1}^L \frac{\partial y^c}{\partial A_i^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The resulting 1D attribution map is linearly interpolated to match the original 256-sample window and mapped via the `inferno` colormap.

### 3. Autonomic HRV Equations
* **SDNN (Standard Deviation of NN Intervals):**
  $$\text{SDNN} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (RR_i - \overline{RR})^2}$$
* **RMSSD (Root Mean Square of Successive Differences):**
  $$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$$

---

## 🗂️ Repository Structure

```text
ecg-arrhythmia-classification/
├── app.py                         # Unified Streamlit Clinical Diagnostic Studio
├── model.py                       # 1D PyTorch Arrhythmia Classifier Architecture
├── config.py                      # AAMI EC57 class mappings & clinical constants
├── pan_tompkins.py                # Standalone Pan-Tompkins QRS detection engine
├── generate_sample_patient_ecg.py # Synthetic continuous ECG trace generator (PVC/SVPB)
├── requirements.txt               # Production dependencies
├── ecg_arrhythmia_best.pth        # Trained PyTorch model checkpoint
├── X_test.npy / y_test.npy        # AAMI DS2 inter-patient test cohort data
└── README.md                      # Clinical & system documentation

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/junaidmir1212/ecg-arrhythmia-classification.git](https://github.com/junaidmir1212/ecg-arrhythmia-classification.git)
cd ecg-arrhythmia-classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## 📊 AAMI EC57 Inter-Patient Benchmark

Model performance is evaluated against the MIT-BIH Arrhythmia Database strictly partitioned according to the **de Chazal et al. (2004)** inter-patient protocol to prevent data leakage:

| Class Identifier | Clinical Description | Train/Val ($DS1$) | Test ($DS2$) | Primary Feature Focus |
| :---: | :--- | :---: | :---: | :--- |
| **N** | Normal Sinus / Bundle Branches | 45,844 | 44,259 | Baseline PR, QRS, ST-T continuity |
| **S** | Supraventricular Ectopic (SVPB) | 943 | 1,836 | Premature P-wave / compensatory interval |
| **V** | Ventricular Ectopic (PVC) | 3,788 | 3,220 | Broad, bizarre QRS / inverted T-wave |
| **F** | Fusion (Ventricular + Normal) | 414 | 388 | Hybrid morphology |
| **Q** | Paced / Unclassifiable | 8 | 7 | Sharp stimulus pacing artifacts |

---

## 📑 Clinical Audit Report Sample

The automated PDF report outputs:
1. **Patient Recording Summary**: Total beats, sampling frequency, recording duration, mean HR.
2. **Autonomic HRV Profile**: SDNN, RMSSD, and pNN50 metrics.
3. **Ischemia & Infarction Assessment**: Measured $J+60\text{ ms}$ voltage shifts relative to the $PR$ baseline.
4. **AAMI EC57 Distribution**: Itemized count and clinical prevalence percentages.
5. **Deep Feature Attribution**: Embedded high-resolution 1D Grad-CAM plot highlighting aberrant intervals.

---

## ⚖️ Clinical Disclaimer

> **Notice:** This software is an experimental, edge-optimized diagnostic decision-support model developed for biomedical research and academic auditing under the AAMI EC57 benchmark. It is not an FDA-cleared diagnostic medical device and should not replace primary evaluation by a qualified medical practitioner or certified electrophysiologist.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.