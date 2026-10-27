# app.py
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import ECGArrhythmiaClassifier
from config import CLASS_NAMES

st.set_page_config(
    page_title="ECG Arrhythmia AI Diagnostic Studio",
    page_icon="🫀",
    layout="wide"
)

# --- 1. Model & Data Caching ---
@st.cache_resource
def load_model():
    import os
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
    import os
    if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
        return np.load('X_test.npy'), np.load('y_test.npy')
    elif os.path.exists('sample_X_test.npy') and os.path.exists('sample_y_test.npy'):
        return np.load('sample_X_test.npy'), np.load('sample_y_test.npy')
    
    # Extract real heartbeats directly from patient record 100
    import wfdb
    from scipy.signal import butter, filtfilt
    
    rec_path = 'mitdb/100'
    if not os.path.exists(f"{rec_path}.dat"):
        os.makedirs('mitdb', exist_ok=True)
        wfdb.dl_files('mitdb', 'mitdb', ['100.dat', '100.hea', '100.atr'])
        
    record = wfdb.rdrecord(rec_path, channels=[0])
    ann = wfdb.rdann(rec_path, 'atr')
    raw_signal = record.p_signal[:, 0]
    
    # 0.5 - 45 Hz Butterworth filter
    nyq = 0.5 * 360
    b, a = butter(3, [0.5 / nyq, 45.0 / nyq], btype='band')
    filtered = filtfilt(b, a, raw_signal)
    
    beats, labels = [], []
    for r_peak, sym in zip(ann.sample[:300], ann.symbol[:300]):
        start, end = r_peak - 128, r_peak + 128
        if start >= 0 and end < len(filtered):
            beat = filtered[start:end]
            std = np.std(beat)
            beat = (beat - np.mean(beat)) / (std if std > 1e-6 else 1e-6)
            beats.append(beat)
            # Patient 100 primarily has Normal (0) with occasional PVC / VEB (2)
            labels.append(2 if sym in ['V', 'E'] else 0)
            
    X = np.expand_dims(np.array(beats, dtype=np.float32), axis=1)
    y = np.array(labels, dtype=np.int64)
    return X, y
# --- 2. 1D Grad-CAM Engine ---
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

# --- 3. UI Application ---
def main():
    st.title("🫀 Real-Time ECG Arrhythmia Diagnostics & Interpretability")
    st.markdown(
        "Interactive testing interface using the **MIT-BIH Arrhythmia inter-patient test cohort (DS2)**. "
        "Select a heartbeat to run real-time inference and generate **1D Grad-CAM diagnostic attribution heatmaps**."
    )

    try:
        model, device = load_model()
        X_test, y_test = load_test_data()
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure preprocess.py and train.py have been executed.")
        return

    # Sidebar: Beat Selection Controls
    st.sidebar.header("ECG Sample Selector")
    
    # Filter by true ground truth class
    selected_class_label = st.sidebar.selectbox(
        "Filter by Ground-Truth Class:",
        options=list(range(len(CLASS_NAMES))),
        format_func=lambda idx: CLASS_NAMES[idx]
    )
    
    available_indices = np.where(y_test == selected_class_label)[0]
    st.sidebar.info(f"Available samples in DS2 for this class: **{len(available_indices)}**")
    
    # Dynamic key prevents duplicate element IDs; safe conditional prevents min==max error
    if len(available_indices) > 1:
        sample_sub_idx = st.sidebar.slider(
            "Select Sample Index:",
            min_value=0,
            max_value=len(available_indices) - 1,
            value=0,
            key=f"sample_slider_{beat_class}"
        )
    else:
        sample_sub_idx = 0
        
    beat_idx = available_indices[sample_sub_idx]

    # Inference
    beat_signal = X_test[beat_idx, 0]
    input_tensor = torch.from_numpy(X_test[beat_idx:beat_idx+1]).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    true_class = y_test[beat_idx]

    # Compute Grad-CAM
    cam_engine = GradCAM1D(model, target_layer=model.feature_extractor[8])
    # Compute attribution for predicted class
    cam_weights = cam_engine.generate(input_tensor, class_idx=pred_class)

    # Main Display: Layout Columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Diagnostic Assessment")
        
        # Diagnostic Match Card
        is_correct = (pred_class == true_class)
        if is_correct:
            st.success(f"**Predicted:** {CLASS_NAMES[pred_class]}")
        else:
            st.error(f"**Predicted:** {CLASS_NAMES[pred_class]}")
            
        st.markdown(f"**Ground Truth:** {CLASS_NAMES[true_class]}")
        st.markdown(f"**Confidence:** `{probs[pred_class]*100:.2f}%`")
        
        st.write("---")
        st.markdown("#### Class Probabilities")
        for idx, (c_name, p) in enumerate(zip(CLASS_NAMES, probs)):
            st.write(f"**{c_name}**")
            st.progress(float(p))
            st.caption(f"{p*100:.2f}%")

    with col2:
        st.subheader("1D Grad-CAM Diagnostic Waveform Attribution")
        
        time_ms = np.linspace(-355, 355, len(beat_signal))
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
        )
        
        # Upper plot: ECG signal with Grad-CAM colormap
        ax1.plot(time_ms, beat_signal, color='#2c3e50', lw=1.6, label='Normalized ECG')
        scatter = ax1.scatter(time_ms, beat_signal, c=cam_weights, cmap='inferno', s=16, zorder=3)
        ax1.set_ylabel("Normalized Voltage")
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(loc='upper right')
        cbar = plt.colorbar(scatter, ax=ax1, orientation='vertical')
        cbar.set_label('Attribution (Attention Weight)')
        
        # Lower plot: Heatmap ribbon
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

if __name__ == '__main__':
    main()