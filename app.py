# app.py
import os
import io
import tempfile
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from fpdf import FPDF

from model import ECGArrhythmiaClassifier
from config import CLASS_NAMES

st.set_page_config(
    page_title="ECG Arrhythmia AI Diagnostic Studio",
    page_icon="🫀",
    layout="wide"
)


# =====================================================================
# 1. Embedded Pan-Tompkins Autonomous QRS Detector
# =====================================================================
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


# =====================================================================
# 2. HRV and ST Deviation Clinical Measurement Engines
# =====================================================================
def compute_hrv_metrics(r_peaks, fs=360):
    if len(r_peaks) < 2:
        return None, None, None
    
    r_peaks = np.array(r_peaks, dtype=np.float32)
    
    # R-R intervals in milliseconds
    rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
    rr_times_sec = r_peaks[1:] / fs

    mean_rr = float(np.mean(rr_intervals_ms))
    mean_hr = float(60000.0 / mean_rr) if mean_rr > 0 else 0.0
    sdnn = float(np.std(rr_intervals_ms, ddof=1)) if len(rr_intervals_ms) > 1 else 0.0
    
    # Successive differences
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = float(nn50 / len(successive_diffs) * 100.0) if len(successive_diffs) > 0 else 0.0

    metrics = {
        "mean_hr": mean_hr,
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50
    }
    return metrics, rr_times_sec, rr_intervals_ms
    
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (nn50 / len(successive_diffs) * 100.0) if len(successive_diffs) > 0 else 0.0

    metrics = {
        "mean_hr": mean_hr,
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50
    }
    return metrics, rr_times_sec, rr_intervals_ms


def measure_st_deviation(beat_raw, r_idx_local=128, fs=360):
    pr_start = max(0, r_idx_local - int(0.080 * fs))
    pr_end = max(0, r_idx_local - int(0.040 * fs))
    isoelectric_voltage = np.median(beat_raw[pr_start:pr_end]) if pr_end > pr_start else beat_raw[0]

    qrs_search_start = r_idx_local + int(0.025 * fs)
    qrs_search_end = min(len(beat_raw), r_idx_local + int(0.090 * fs))
    s_trough = qrs_search_start + np.argmin(beat_raw[qrs_search_start:qrs_search_end])
    
    diffs = np.diff(beat_raw)
    j_idx = s_trough
    for i in range(s_trough, min(len(diffs), s_trough + int(0.040 * fs))):
        if abs(diffs[i]) < 0.005:
            j_idx = i
            break

    st_sample_idx = min(len(beat_raw) - 1, j_idx + int(0.060 * fs))
    st_voltage = beat_raw[st_sample_idx]
    st_deviation_mv = float(st_voltage - isoelectric_voltage)

    if st_deviation_mv >= 0.10:
        diagnosis = "STEMI Alert (ST-Elevation)"
        severity = "CRITICAL"
        color = "#e74c3c"
    elif st_deviation_mv <= -0.05:
        diagnosis = "Ischemia Alert (ST-Depression)"
        severity = "HIGH"
        color = "#e67e22"
    else:
        diagnosis = "Normal ST Contour"
        severity = "BENIGN"
        color = "#27ae60"

    return {
        "st_deviation_mv": st_deviation_mv,
        "isoelectric_mv": isoelectric_voltage,
        "j_idx": j_idx,
        "st_idx": st_sample_idx,
        "diagnosis": diagnosis,
        "severity": severity,
        "color": color
    }

# =====================================================================
# 12-Lead ECG Processing & Multi-Lead Ischemia Engines
# =====================================================================
STANDARD_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

ANATOMICAL_LEAD_GROUPS = {
    "Inferior Wall (RCA / LCx)": ['II', 'III', 'aVF'],
    "Septal Wall (LAD)": ['V1', 'V2'],
    "Anterior Wall (LAD)": ['V3', 'V4'],
    "Lateral Wall (LCx / Diagonal)": ['I', 'aVL', 'V5', 'V6'],
    "Reciprocal / Cavitary": ['aVR']
}

def parse_ecg_file(uploaded_file):
    """
    Parses CSV, TXT, or NPY files into a structured dictionary:
    Returns:
        dict: {
            'leads': dict mapping lead_name -> 1D numpy array,
            'lead_names': list of available leads,
            'is_12_lead': bool
        }
    """
    if uploaded_file.name.endswith(".npy"):
        data = np.load(uploaded_file)
        if data.ndim == 1:
            return {'leads': {'Lead II': data.astype(np.float32)}, 'lead_names': ['Lead II'], 'is_12_lead': False}
        elif data.ndim == 2:
            if data.shape[0] == 12:
                leads_dict = {STANDARD_12_LEADS[i]: data[i].astype(np.float32) for i in range(12)}
                return {'leads': leads_dict, 'lead_names': STANDARD_12_LEADS, 'is_12_lead': True}
            elif data.shape[1] == 12:
                leads_dict = {STANDARD_12_LEADS[i]: data[:, i].astype(np.float32) for i in range(12)}
                return {'leads': leads_dict, 'lead_names': STANDARD_12_LEADS, 'is_12_lead': True}
            else:
                return {'leads': {'Lead 0': data[:, 0].astype(np.float32)}, 'lead_names': ['Lead 0'], 'is_12_lead': False}

    # CSV / TXT Parsing
    df = pd.read_csv(uploaded_file)
    cols = [str(c).strip() for c in df.columns]
    matched_leads = [lead for lead in STANDARD_12_LEADS if lead in cols]
    
    if len(matched_leads) >= 12:
        leads_dict = {lead: df[lead].values.astype(np.float32) for lead in STANDARD_12_LEADS}
        return {'leads': leads_dict, 'lead_names': STANDARD_12_LEADS, 'is_12_lead': True}
    elif df.shape[1] == 12:
        leads_dict = {STANDARD_12_LEADS[i]: df.iloc[:, i].values.astype(np.float32) for i in range(12)}
        return {'leads': leads_dict, 'lead_names': STANDARD_12_LEADS, 'is_12_lead': True}
    else:
        return {'leads': {'Lead II': df.iloc[:, 0].values.astype(np.float32)}, 'lead_names': ['Lead II'], 'is_12_lead': False}


def localize_12lead_ischemia(leads_dict, r_peak_idx, fs=360):
    st_summary = {}
    contiguous_alerts = []

    for lead_name in STANDARD_12_LEADS:
        if lead_name not in leads_dict:
            continue
        sig = leads_dict[lead_name]
        if r_peak_idx - 128 >= 0 and r_peak_idx + 128 < len(sig):
            beat = sig[r_peak_idx - 128 : r_peak_idx + 128]
            st_res = measure_st_deviation(beat, r_idx_local=128, fs=fs)
            st_summary[lead_name] = st_res

    for territory, group in ANATOMICAL_LEAD_GROUPS.items():
        elevated_in_group = [l for l in group if l in st_summary and st_summary[l]["st_deviation_mv"] >= 0.10]
        if len(elevated_in_group) >= 2:
            contiguous_alerts.append(f"{territory} (Elevated in: {', '.join(elevated_in_group)})")

    return st_summary, contiguous_alerts

# =====================================================================
# 3. Clinical PDF Report Generator
# =====================================================================
class ClinicalECGReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(26, 82, 118)
        self.cell(0, 10, "CARDIOVASCULAR AI DIAGNOSTIC AUDIT REPORT", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Automated Continuous Rhythm & Morphological Attribution Audit", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.line(10, 26, 200, 26)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"ECG Arrhythmia AI Diagnostic Studio | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}", align="C")


def create_clinical_pdf(metrics_dict, anomalies_list, hrv_metrics=None, st_data=None, fig_anomaly=None):
    pdf = ClinicalECGReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Study Metadata
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 7, "1. Patient Trace Assessment Summary", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 6, f"Recording Length: {metrics_dict['duration_sec']:.1f} s ({metrics_dict['total_samples']} samples)", border=1)
    pdf.cell(95, 6, f"Sampling Frequency: {metrics_dict['fs']} Hz", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(95, 6, f"Total QRS Detected: {metrics_dict['total_beats']}", border=1)
    pdf.cell(95, 6, f"Mean Heart Rate: {metrics_dict['mean_hr']:.0f} bpm", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Autonomic Tone & HRV
    if hrv_metrics is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 7, "2. Autonomic Assessment & HRV Biomarkers", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 243, 244)
        pdf.cell(50, 6, "Biomarker", border=1, fill=True)
        pdf.cell(40, 6, "Calculated Value", border=1, fill=True)
        pdf.cell(100, 6, "Physiological Indication", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        hrv_rows = [
            ("Mean R-R Interval", f"{hrv_metrics['mean_rr']:.0f} ms", "Baseline cardiac cycle duration"),
            ("SDNN", f"{hrv_metrics['sdnn']:.1f} ms", "Global autonomic tone & circadian variability"),
            ("RMSSD", f"{hrv_metrics['rmssd']:.1f} ms", "Parasympathetic / vagal regulatory modulation"),
            ("pNN50", f"{hrv_metrics['pnn50']:.1f} %", "Percentage of successive differences > 50 ms")
        ]
        for metric_name, val_str, indication in hrv_rows:
            pdf.cell(50, 6, metric_name, border=1)
            pdf.cell(40, 6, val_str, border=1)
            pdf.cell(100, 6, indication, border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Ischemia / ST Deviation
    if st_data is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 7, "3. Ischemic Injury Assessment (ST-Segment Deviation)", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.cell(60, 6, f"Measured ST Shift: {st_data['st_deviation_mv']:+.2f} mV", border=1)
        pdf.cell(50, 6, f"Status: {st_data['severity']}", border=1)
        pdf.cell(80, 6, f"Interpretation: {st_data['diagnosis']}", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # AAMI EC57 Beat Classification Distribution
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 7, "4. AAMI EC57 Rhythm Triage Counts", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 243, 244)
    pdf.cell(60, 6, "Class Category", border=1, fill=True)
    pdf.cell(35, 6, "Count", border=1, fill=True)
    pdf.cell(35, 6, "Prevalence (%)", border=1, fill=True)
    pdf.cell(60, 6, "Clinical Severity", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    classes_info = [
        ("N (Normal Sinus)", metrics_dict['n_count'], "Baseline Conduction"),
        ("V (Ventricular Ectopic / PVC)", metrics_dict['v_count'], "High Priority (Malignant Potential)"),
        ("S (Supraventricular Ectopic)", metrics_dict['s_count'], "Moderate Priority"),
        ("F / Q (Fusion / Unclassifiable)", metrics_dict['fq_count'], "Artifact / Complex Conduction")
    ]
    total_b = max(metrics_dict['total_beats'], 1)
    for name, cnt, sev in classes_info:
        pct = (cnt / total_b) * 100
        pdf.cell(60, 6, name, border=1)
        pdf.cell(35, 6, str(cnt), border=1)
        pdf.cell(35, 6, f"{pct:.1f}%", border=1)
        pdf.cell(60, 6, sev, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Anomaly Attribution & Grad-CAM Plot
    if fig_anomaly is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 7, "5. Deep Feature Attribution (1D Grad-CAM Diagnostic Focus)", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, "The plot below illustrates salient activations driving the model's classification. Peaks highlighted in inferno colormap isolate aberrant intervals (e.g. widened QRS or ST deviations).")
        pdf.ln(1)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig_anomaly.savefig(tmpfile.name, dpi=200, bbox_inches="tight")
            pdf.image(tmpfile.name, x=15, w=180)
        pdf.ln(3)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, "Notice: This automated report is generated by an edge-optimized neural decision-support model under the AAMI EC57 inter-patient benchmark. It is intended to assist medical professionals by prioritizing anomalies and should not replace primary clinical evaluation by a certified electrophysiologist.")

    return pdf.output()


# =====================================================================
# 4. Model and Benchmark Data Caching
# =====================================================================
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


# =====================================================================
# 5. 1D Grad-CAM Diagnostic Waveform Attribution Engine
# =====================================================================
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


# =====================================================================
# 6. Main Streamlit Diagnostic Studio
# =====================================================================
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

    # -----------------------------------------------------------------
    # TAB 1: Single Beat Inspector
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # TAB 2: Continuous Patient ECG Scanner
    # -----------------------------------------------------------------
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
        leads_dict = {}
        is_12_lead = False
        fs = 360

        if uploaded_file is not None:
            parsed = parse_ecg_file(uploaded_file)
            leads_dict = parsed['leads']
            is_12_lead = parsed['is_12_lead']
            primary_lead = 'II' if 'II' in leads_dict else parsed['lead_names'][0]
            raw_signal = leads_dict[primary_lead]
        elif demo_btn:
            import wfdb
            rec_path = 'mitdb/100'
            if not os.path.exists(f"{rec_path}.dat"):
                os.makedirs('mitdb', exist_ok=True)
                wfdb.dl_files('mitdb', 'mitdb', ['100.dat', '100.hea', '100.atr'])
            rec = wfdb.rdrecord(rec_path, sampfrom=0, sampto=3600, channels=[0])
            raw_signal = rec.p_signal[:, 0].astype(np.float32)
            leads_dict = {'Lead II': raw_signal}
            is_12_lead = False

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
                
                # --- Continuous Rhythm Timeline with Click Selection ---
                st.write("---")
                st.subheader("📈 Continuous Rhythm Timeline (Click Any Peak to Inspect)")
                
                time_strip_sec = np.arange(len(filtered_signal)) / fs
                peak_times_sec = np.array(valid_peaks) / fs
                peak_voltages = filtered_signal[valid_peaks]

                color_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c", 3: "#9b59b6", 4: "#95a5a6"}
                marker_colors = [color_map.get(p, "#34495e") for p in batch_preds]
                marker_labels = [f"Beat #{i}: {CLASS_NAMES[p]} ({batch_probs[i][p]*100:.1f}%)" 
                                 for i, p in enumerate(batch_preds)]

                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(
                    x=time_strip_sec,
                    y=filtered_signal,
                    mode='lines',
                    line=dict(color='#7f8c8d', width=1),
                    name='ECG Lead MLII',
                    hoverinfo='skip'
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=peak_times_sec,
                    y=peak_voltages,
                    mode='markers',
                    marker=dict(size=10, color=marker_colors, symbol='circle'),
                    text=marker_labels,
                    hoverinfo='text',
                    name='R-Peaks',
                    customdata=np.arange(len(valid_peaks))
                ))
                fig_timeline.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=10, b=20),
                    xaxis=dict(title="Time (seconds)", showgrid=True),
                    yaxis=dict(title="Voltage (mV)", showgrid=True),
                    showlegend=False,
                    clickmode='event+select'
                )

                timeline_event = st.plotly_chart(
                    fig_timeline, 
                    use_container_width=True, 
                    on_select="rerun", 
                    selection_mode="points"
                )

                selected_beat_idx = 0
                if timeline_event and "points" in timeline_event.get("selection", {}) and len(timeline_event["selection"]["points"]) > 0:
                    clicked_point = timeline_event["selection"]["points"][0]
                    if clicked_point.get("curve_number") == 1:
                        selected_beat_idx = int(clicked_point["point_index"])
                elif len(abnormal_idxs) > 0:
                    selected_beat_idx = int(abnormal_idxs[0])

                inspect_idx = st.selectbox(
                    "Selected Heartbeat for Deep Diagnostic Audit:",
                    options=list(range(len(valid_beats))),
                    index=selected_beat_idx,
                    format_func=lambda idx: f"Beat #{idx} at {(valid_peaks[idx]/fs):.2f}s | Predicted: {CLASS_NAMES[batch_preds[idx]]} ({batch_probs[idx][batch_preds[idx]]*100:.1f}%)"
                )

                # --- Heart Rate Variability (HRV) & R-R Tachogram ---
                hrv_metrics, rr_times, rr_ms = compute_hrv_metrics(valid_peaks, fs=fs)

                if hrv_metrics is not None:
                    st.write("---")
                    st.subheader("🫀 Autonomic Assessment: Heart Rate Variability (HRV)")
                    
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Mean R-R Interval", f"{hrv_metrics['mean_rr']:.0f} ms")
                    h2.metric("SDNN (Global Variability)", f"{hrv_metrics['sdnn']:.1f} ms")
                    h3.metric("RMSSD (Parasympathetic)", f"{hrv_metrics['rmssd']:.1f} ms")
                    h4.metric("pNN50", f"{hrv_metrics['pnn50']:.1f} %")

                    fig_tachogram = go.Figure()
                    fig_tachogram.add_trace(go.Scatter(
                        x=rr_times,
                        y=rr_ms,
                        mode='lines+markers',
                        line=dict(color='#2980b9', width=1.5),
                        marker=dict(size=5, color='#e74c3c'),
                        name='R-R Interval',
                        hovertemplate='Time: %{x:.2f}s<br>R-R: %{y:.0f} ms<extra></extra>'
                    ))
                    fig_tachogram.add_hline(
                        y=hrv_metrics['mean_rr'], 
                        line_dash="dash", 
                        line_color="#27ae60",
                        annotation_text=f"Mean R-R: {hrv_metrics['mean_rr']:.0f} ms",
                        annotation_position="bottom right"
                    )
                    fig_tachogram.update_layout(
                        title="R-R Interval Tachogram (Beat-to-Beat Progression)",
                        height=220,
                        margin=dict(l=20, r=20, t=35, b=20),
                        xaxis=dict(title="Time along Recording (s)", showgrid=True),
                        yaxis=dict(title="R-R Interval (ms)", showgrid=True),
                        showlegend=False
                    )
                    st.plotly_chart(fig_tachogram, use_container_width=True)

                # --- 1D Grad-CAM on Inspected Beat ---
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

                # --- ST-Segment Morphology & Ischemia Assessment ---
                raw_beat_unnorm = filtered_signal[valid_peaks[inspect_idx] - 128 : valid_peaks[inspect_idx] + 128]
                st_data = measure_st_deviation(raw_beat_unnorm, r_idx_local=128, fs=fs)

                st.write("---")
                st.subheader("🩺 Myocardial Ischemia & Infarction Audit (ST Deviation)")

                col_st1, col_st2, col_st3 = st.columns(3)
                col_st1.metric(
                    "ST Deviation (J+60ms)",
                    f"{st_data['st_deviation_mv']:+.2f} mV",
                    delta=f"{st_data['st_deviation_mv'] * 10:+.1f} mm deflection",
                    delta_color="inverse" if abs(st_data['st_deviation_mv']) >= 0.05 else "off"
                )
                col_st2.metric("Isoelectric Baseline", f"{st_data['isoelectric_mv']:.2f} mV")
                col_st3.markdown(
                    f"**Pathology Assessment:**  \n<span style='color:{st_data['color']}; font-weight:bold; font-size:18px;'>{st_data['diagnosis']}</span>", 
                    unsafe_allow_html=True
                )

                fig_st, ax_st = plt.subplots(figsize=(10, 2.5))
                t_samples = np.arange(len(raw_beat_unnorm))
                ax_st.plot(t_samples, raw_beat_unnorm, color='#2c3e50', lw=1.5, label='Filtered ECG (mV)')
                ax_st.axhline(st_data['isoelectric_mv'], color='gray', linestyle='--', lw=1, label='Isoelectric (PR)')
                ax_st.scatter(st_data['j_idx'], raw_beat_unnorm[st_data['j_idx']], color='#3498db', s=40, zorder=5, label='J-Point')
                ax_st.scatter(st_data['st_idx'], raw_beat_unnorm[st_data['st_idx']], color=st_data['color'], s=60, zorder=5, label=f"J+60ms ({st_data['st_deviation_mv']:+.2f} mV)")
                ax_st.set_ylabel("Voltage (mV)")
                ax_st.set_xlabel("Sample Point (Relative to R-Peak at 128)")
                ax_st.grid(True, linestyle=':', alpha=0.5)
                ax_st.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_st)
                plt.close(fig_st)

# --- 12-Lead Standard Clinical ECG Montage Grid ---
                if is_12_lead:
                    st.write("---")
                    st.subheader("📋 12-Lead Diagnostic Electrocardiogram Grid")
                    
                    st_12lead_res, territory_alerts = localize_12lead_ischemia(
                        leads_dict, valid_peaks[inspect_idx], fs=fs
                    )
                    
                    if len(territory_alerts) > 0:
                        st.error(f"🚨 **Contiguous STEMI Pattern Detected:** {', '.join(territory_alerts)}")
                    else:
                        st.info("No contiguous ST-elevation criteria met across standard anatomical lead groups.")

                    col_leads = [
                        ['I', 'aVR', 'V1', 'V4'],
                        ['II', 'aVL', 'V2', 'V5'],
                        ['III', 'aVF', 'V3', 'V6']
                    ]
                    grid_cols = st.columns(3)

                    for c_idx, group in enumerate(col_leads):
                        with grid_cols[c_idx]:
                            for l_name in group:
                                if l_name in leads_dict and l_name in st_12lead_res:
                                    fig_lead, ax_l = plt.subplots(figsize=(4, 1.2))
                                    beat_l = leads_dict[l_name][valid_peaks[inspect_idx]-128 : valid_peaks[inspect_idx]+128]
                                    st_shift = st_12lead_res[l_name]['st_deviation_mv']
                                    
                                    lead_color = '#e74c3c' if st_shift >= 0.10 else ('#e67e22' if st_shift <= -0.05 else '#2c3e50')
                                    ax_l.plot(beat_l, color=lead_color, lw=1.2)
                                    ax_l.set_title(f"Lead {l_name} (ST: {st_shift:+.2f} mV)", fontsize=8, pad=2)
                                    ax_l.set_xticks([])
                                    ax_l.set_yticks([])
                                    ax_l.grid(True, linestyle=':', alpha=0.3)
                                    st.pyplot(fig_lead)
                                    plt.close(fig_lead)
                                    

                # --- PDF Generation Action ---
                st.write("---")
                st.subheader("📄 Clinical Decision-Support Export")

                mean_hr_val = (len(valid_peaks) / (duration_sec / 60.0)) if duration_sec > 0 else 0
                metrics_payload = {
                    "duration_sec": duration_sec,
                    "total_samples": len(raw_signal),
                    "fs": fs,
                    "total_beats": len(valid_beats),
                    "mean_hr": mean_hr_val,
                    "n_count": int(np.sum(batch_preds == 0)),
                    "v_count": int(np.sum(batch_preds == 2)),
                    "s_count": int(np.sum(batch_preds == 1)),
                    "fq_count": int(np.sum((batch_preds == 3) | (batch_preds == 4))),
                }

                pdf_bytes = create_clinical_pdf(
                    metrics_dict=metrics_payload,
                    anomalies_list=abnormal_idxs,
                    hrv_metrics=hrv_metrics,
                    st_data=st_data,
                    fig_anomaly=fig
                )

                st.download_button(
                    label="📥 Download Clinical Audit Report (PDF)",
                    data=bytes(pdf_bytes),
                    file_name=f"ECG_Clinical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("No complete heartbeats could be segmented from the provided signal duration.")


if __name__ == '__main__':
    main()