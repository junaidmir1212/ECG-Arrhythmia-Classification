# 🫀 Interpretable & Edge-Optimized ECG Arrhythmia Diagnostics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-subdomain.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Edge_Optimized-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Standard](https://img.shields.io/badge/Evaluation-AAMI_EC57-brightgreen)](#clinical-benchmark--evaluation)

An end-to-end biomedical machine learning system for real-time cardiac arrhythmia detection from continuous electrocardiogram (ECG) biosignals using the MIT-BIH Arrhythmia Database.

This project enforces the clinical **AAMI EC57 standard** using the strict **inter-patient Chazal et al. partition** (DS1 for training, DS2 for evaluation) to eliminate patient identity leakage. It integrates **autonomous Pan-Tompkins QRS peak detection**, multi-tier Explainable AI (**1D Grad-CAM**, **Integrated Gradients**, and **Counterfactual Minimal Repairs**), **ONNX Runtime edge acceleration**, and an interactive **Streamlit Diagnostic Studio**.

---

## 🌟 Key Upgrades & Technical Architecture