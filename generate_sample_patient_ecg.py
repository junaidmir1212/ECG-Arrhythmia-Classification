# generate_sample_patient_ecg.py
import numpy as np
import pandas as pd

def generate_patient_ecg(filename="patient_ecg.csv", duration_sec=10, fs=360):
    total_samples = int(duration_sec * fs)
    continuous_signal = np.zeros(total_samples, dtype=np.float32)

    # 1. Morphological waveform templates
    t = np.linspace(-1, 1, 256)
    
    # Normal sinus beat template: P-wave, sharp QRS, upright T-wave
    normal_beat = (
        0.15 * np.exp(-50 * (t + 0.35)**2)   # P wave
        + 1.0 * np.exp(-150 * t**2)           # R peak
        - 0.20 * np.exp(-150 * (t + 0.05)**2) # Q dip
        - 0.25 * np.exp(-150 * (t - 0.05)**2) # S dip
        + 0.30 * np.exp(-30 * (t - 0.35)**2)  # T wave
    )
    
    # Premature Ventricular Contraction (PVC / V) template:
    # Wide, aberrant, inverted deep QRS complex with broad compensatory T-wave
    pvc_beat = (
        -1.4 * np.exp(-35 * t**2)             # Widened inverted QRS
        + 0.45 * np.exp(-20 * (t - 0.40)**2)  # Discordant elevated T-wave
    )

    # 2. Sequence rhythm timing (10-second sequence with ~12 beats)
    # Beats 5 and 9 will simulate premature ventricular ectopic occurrences
    beat_schedule = [
        (250, "N"),
        (550, "N"),
        (850, "N"),
        (1150, "N"),
        (1400, "V"),  # Early ectopic beat (PVC)
        (1800, "N"),  # Compensatory pause followed by normal sinus
        (2100, "N"),
        (2400, "N"),
        (2660, "V"),  # Second ectopic beat (PVC)
        (3050, "N"),
        (3350, "N")
    ]

    # 3. Splice beats into continuous trace
    for r_center, beat_type in beat_schedule:
        template = pvc_beat if beat_type == "V" else normal_beat
        start = r_center - 128
        end = r_center + 128
        if start >= 0 and end < total_samples:
            continuous_signal[start:end] += template

    # 4. Inject physiological artifacts
    # Baseline wander (respiration ~0.25 Hz)
    time_sec = np.linspace(0, duration_sec, total_samples)
    baseline_wander = 0.08 * np.sin(2 * np.pi * 0.25 * time_sec)
    
    # High-frequency physiological / EMG noise
    high_freq_noise = np.random.normal(0, 0.02, total_samples).astype(np.float32)
    
    continuous_signal += baseline_wander + high_freq_noise

    # 5. Export single-column Lead MLII continuous signal to CSV
    df = pd.DataFrame(continuous_signal)
    df.to_csv(filename, index=False, header=False)
    print(f"Generated '{filename}' successfully: {total_samples} samples (~{duration_sec}s at {fs}Hz).")

if __name__ == "__main__":
    generate_patient_ecg()