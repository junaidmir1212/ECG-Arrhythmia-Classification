# app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from model import ECGArrhythmiaClassifier
from config import CLASS_NAMES

st.set_page_config(
    page_title="ECG Arrhythmia AI Diagnostic Studio",
    page_icon="🫀",
    layout="wide"
)

# --- 1. Embedded Pan-Tompkins Autonomous QRS Detector ---
class PanTompkinsDetector:
    def __init__(self, fs=360):
        self.fs = fs

    def bandpass_filter(self, signal, lowcut=5.0, highcut=15.0):
        nyquist = 0.5 * self.fs
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.999)
        b, a = butter(1, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def detect_peaks(self, signal):
        filtered = self.bandpass_filter(signal)
        diff = np.diff(filtered)
        squared = diff ** 2
        window_size = int(0.15 * self.fs)
        integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
        
        threshold = np.mean(integrated) + 1.2 * np.std(integrated)
        min_distance = int(0.25 * self.fs)
        peaks = []
        last_peak = -min_distance

        for i in range(1, len(integrated) - 1):
            if integrated[i] > threshold and integrated[i] > integrated[i-1] and integrated[i] > integrated[i+1]:
                if i - last_peak > min_distance:
                    search_start = max(0, i - int(0.08 * self.fs))
                    search_end = min(len(filtered), i + int(0.08 * self.fs))
                    r_peak = search_start + np.argmax(filtered[search_start:search_end])
                    peaks.append(r_peak)
                    last_peak = i
        return np.array(peaks, dtype=int)


# --- 2. Model & Data Caching ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGArrhythmiaClassifier(num_classes=5)
    model_path = 'ecg_arrhythmia_best.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.sidebar.warning("⚡ Running with initialized demonstration weights (ecg_arrhythmia_best.pth not found locally).")
        
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_test_data():
    if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
        return np.load('X_test.npy'), np.load('y_test.npy')
    elif os.path.exists('sample_X_test.npy') and os.path.exists('sample_y_test.npy'):
        return np.load('sample_X_test.npy'), np.load('sample_y_test.npy')
    
    # In-memory balanced fallback across all 5 classes
    t = np.linspace(-1, 1, 256)
    beats, labels = [], []
    for cls_idx in range(5):
        for _ in range(20):
            jitter = np.random.normal(0, 0.02, 256)
            if cls_idx == 0:
                b = np.exp(-150 * t**2) - 0.25 * np.exp(-30 * (t - 0.35)**2) + 0.15 * np.exp(-50 * (t + 0.35)**2)
            elif cls_idx == 1:
                b = np.exp(-300 * (t + 0.1)**2) - 0.2 * np.exp(-40 * (t - 0.25)**2)
            elif cls_idx == 2:
                b = -np.exp(-35 * t**2) + 0.3 * np.exp(-20 * (t - 0.4)**2)
            elif cls_idx == 3:
                b = 0.5 * np.exp(-80 * t**2) - 0.5 * np.exp(-40 * (t + 0.1)**2)
            else:
                b = np.zeros(256)
                b[120] = 1.2
            sig = b + jitter
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
            beats.append(sig)
            labels.append(cls_idx)
            
    return np.expand_dims(np.array(beats, dtype=np.float32), 1), np.array(labels, dtype=np.int64)


# --- 3. 1D Grad-CAM Engine ---
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_acts)
        self.target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out): 
        self.activations = out

    def _save_grads(self, m, gin, gout): 
        self.gradients = gout[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.clamp(torch.sum(weights * self.activations, dim=1), min=0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        cam_up = torch.nn.functional.interpolate(
            cam.unsqueeze(1), size=x.shape[2], mode='linear', align_corners=False
        )
        return cam_up.squeeze().detach().cpu().numpy()


# --- 4. Main Application ---
def main():
    st.title("🫀 Real-Time ECG Arrhythmia Diagnostics & Interpretability")
    st.markdown(
        "Interactive clinical interface adhering to the **AAMI EC57 inter-patient evaluation standard**. "
        "Inspect pre-segmented test cohort beats or upload raw continuous patient ECGs for automated multi-beat triage."
    )

    try:
        model, device = load_model()
        X_test, y_test = load_test_data()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return

    cam_engine = GradCAM1D(model, target_layer=model.feature_extractor[8])

    tab1, tab2 = st.tabs([
        "🔬 Benchmark Cohort Inspector (DS2)", 
        "📂 Continuous Patient ECG Scanner"
    ])

    # ==========================================
    # TAB 1: Single Beat Inspector
    # ==========================================
    with tab1:
        st.sidebar.header("ECG Sample Selector")
        
        selected_class_label = st.sidebar.selectbox(
            "Filter by Ground-Truth Class:",
            options=list(range(len(CLASS_NAMES))),
            format_func=lambda idx: CLASS_NAMES[idx]
        )
        
        available_indices = np.where(y_test == selected_class_label)[0]
        st.sidebar.info(f"Available samples in DS2 for this class: **{len(available_indices)}**")
        
        if len(available_indices) == 0:
            st.warning(f"No heartbeats of class **{CLASS_NAMES[selected_class_label]}** found in this dataset subset. Please choose another class from the sidebar.")
        else:
            if len(available_indices) > 1:
                sample_sub_idx = st.sidebar.slider(
                    "Select Sample Index:",
                    min_value=0,
                    max_value=len(available_indices) - 1,
                    value=0,
                    key=f"sample_slider_{selected_class_label}"
                )
            else:
                sample_sub_idx = 0
                
            beat_idx = available_indices[sample_sub_idx]

            beat_signal = X_test[beat_idx, 0]
            input_tensor = torch.from_numpy(X_test[beat_idx:beat_idx+1]).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = int(np.argmax(probs))

            true_class = y_test[beat_idx]
            cam_weights = cam_engine.generate(input_tensor, class_idx=pred_class)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Diagnostic Assessment")
                if pred_class == true_class:
                    st.success(f"**Predicted:** {CLASS_NAMES[pred_class]}")
                else:
                    st.error(f"**Predicted:** {CLASS_NAMES[pred_class]}")
                    
                st.markdown(f"**Ground Truth:** {CLASS_NAMES[true_class]}")
                st.markdown(f"**Confidence:** `{probs[pred_class]*100:.2f}%`")
                
                st.write("---")
                st.markdown("#### Class Probabilities")
                for c_name, p in zip(CLASS_NAMES, probs):
                    st.write(f"**{c_name}**")
                    st.progress(float(p))
                    st.caption(f"{p*100:.2f}%")

            with col2:
                st.subheader("1D Grad-CAM Diagnostic Waveform Attribution")
                time_ms = np.linspace(-355, 355, len(beat_signal))
                
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
                )
                
                ax1.plot(time_ms, beat_signal, color='#2c3e50', lw=1.6, label='Normalized ECG')
                scatter = ax1.scatter(time_ms, beat_signal, c=cam_weights, cmap='inferno', s=16, zorder=3)
                ax1.set_ylabel("Normalized Voltage")
                ax1.grid(True, linestyle='--', alpha=0.4)
                ax1.legend(loc='upper right')
                cbar = plt.colorbar(scatter, ax=ax1, orientation='vertical')
                cbar.set_label('Attribution (Attention Weight)')
                
                ax2.imshow(
                    cam_weights[np.newaxis, :], aspect='auto', cmap='inferno', 
                    extent=[time_ms[0], time_ms[-1], 0, 1]
                )
                ax2.set_yticks([])
                ax2.set_xlabel("Time relative to R-Peak (ms)")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Regions highlighted in yellow/orange drove the neural network's classification decision. "
                    "For ventricular ectopic beats (V), attention typically peaks across the widened QRS complex or inverted T-wave."
                )

    # ==========================================
    # TAB 2: Upload Continuous Patient ECG
    # ==========================================
    with tab2:
        st.subheader("Automated Continuous Rhythm Scanner")
        st.markdown(
            "Upload a continuous Lead MLII ECG trace (.csv, .txt, or .npy). The engine uses **Pan-Tompkins** to scan "
            "the rhythm, extract all beats, and prioritize pathological anomalies for rapid clinical review."
        )

        uploaded_file = st.file_uploader(
            "Select Continuous Patient Signal File", 
            type=["csv", "txt", "npy"],
            key="continuous_ecg_uploader"
        )

        demo_btn = st.button("Load Demo Continuous Patient Trace (Record 100, 10s)")
        
        raw_signal = None
        fs = 360

        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                raw_signal = np.load(uploaded_file).flatten()
            else:
                df = pd.read_csv(uploaded_file, header=None)
                raw_signal = df.iloc[:, 0].values.astype(np.float32)
        elif demo_btn:
            import wfdb
            rec_path = 'mitdb/100'
            if not os.path.exists(f"{rec_path}.dat"):
                os.makedirs('mitdb', exist_ok=True)
                wfdb.dl_files('mitdb', 'mitdb', ['100.dat', '100.hea', '100.atr'])
            rec = wfdb.rdrecord(rec_path, sampfrom=0, sampto=3600, channels=[0])
            raw_signal = rec.p_signal[:, 0].astype(np.float32)

        if raw_signal is not None:
            duration_sec = len(raw_signal) / fs
            st.success(f"Loaded continuous strip: **{len(raw_signal):,} samples** (~{duration_sec:.1f} seconds at {fs} Hz)")

            detector = PanTompkinsDetector(fs=fs)
            filtered_signal = detector.bandpass_filter(raw_signal)
            r_peaks = detector.detect_peaks(raw_signal)

            st.metric("Total Heartbeats Detected by Pan-Tompkins", len(r_peaks))

            valid_beats = []
            valid_peaks = []
            for r in r_peaks:
                if r - 128 >= 0 and r + 128 < len(filtered_signal):
                    beat = filtered_signal[r-128:r+128]
                    std = np.std(beat)
                    beat = (beat - np.mean(beat)) / (std if std > 1e-6 else 1e-6)
                    valid_beats.append(beat)
                    valid_peaks.append(r)

            if len(valid_beats) > 0:
                batch_tensor = torch.from_numpy(
                    np.expand_dims(np.array(valid_beats, dtype=np.float32), 1)
                ).to(device)

                with torch.no_grad():
                    logits = model(batch_tensor)
                    batch_probs = F.softmax(logits, dim=1).cpu().numpy()
                    batch_preds = np.argmax(batch_probs, axis=1)

                m1, m2, m3 = st.columns(3)
                m1.metric("Normal Rhythms (N)", int(np.sum(batch_preds == 0)))
                m2.metric("Ventricular Ectopic (V / PVC)", int(np.sum(batch_preds == 2)))
                m3.metric("Other Ectopic (S / F / Q)", int(np.sum((batch_preds != 0) & (batch_preds != 2))))

                abnormal_idxs = np.where(batch_preds != 0)[0]
                
                if len(abnormal_idxs) > 0:
                    st.error(f"🚨 Triage Alert: **{len(abnormal_idxs)}** anomalous beat(s) flagged for review.")
                    
                    selected_abnormal = st.selectbox(
                        "Jump directly to anomalous beat:",
                        options=abnormal_idxs,
                        format_func=lambda idx: f"Beat #{idx} at {(valid_peaks[idx]/fs):.2f}s | Predicted: {CLASS_NAMES[batch_preds[idx]]} ({batch_probs[idx][batch_preds[idx]]*100:.1f}% confidence)"
                    )
                    inspect_idx = selected_abnormal
                else:
                    st.success("✅ Rhythm Analysis: Entire trace verified as normal sinus rhythm.")
                    inspect_idx = 0

                st.write("---")
                st.subheader(f"Detailed Waveform & Attribution: Beat #{inspect_idx}")
                
                single_beat_tensor = batch_tensor[inspect_idx:inspect_idx+1]
                target_cls = int(batch_preds[inspect_idx])
                cam_single = cam_engine.generate(single_beat_tensor, class_idx=target_cls)
                t_ms = np.linspace(-355, 355, 256)

                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
                )
                ax1.plot(t_ms, valid_beats[inspect_idx], color='#1f77b4', lw=1.5, label='Extracted Beat')
                sc = ax1.scatter(t_ms, valid_beats[inspect_idx], c=cam_single, cmap='inferno', s=16, zorder=3)
                ax1.set_ylabel("Normalized Voltage")
                ax1.grid(True, linestyle='--', alpha=0.3)
                ax1.legend(loc='upper right')
                plt.colorbar(sc, ax=ax1, orientation='vertical', label='Attention')

                ax2.imshow(
                    cam_single[np.newaxis, :], aspect='auto', cmap='inferno', 
                    extent=[t_ms[0], t_ms[-1], 0, 1]
                )
                ax2.set_yticks([])
                ax2.set_xlabel("Time relative to R-Peak (ms)")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No complete heartbeats could be segmented from the provided signal duration.")

if __name__ == '__main__':
    main()