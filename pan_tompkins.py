# pan_tompkins.py
import numpy as np
from scipy.signal import butter, filtfilt

def pan_tompkins_qrs(signal, fs=360):
    """
    Classic Pan-Tompkins QRS detection pipeline:
    1. Bandpass filter (5-15 Hz)
    2. Derivative filter
    3. Squaring
    4. Moving Window Integration (MWI)
    5. Adaptive thresholding
    """
    # 1. Bandpass Filter (5 - 15 Hz to isolate QRS energy)
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)

    # 2. Five-point Derivative: y[n] = (1/8fs) * (2x[n] + x[n-1] - x[n-3] - 2x[n-4])
    derivative = np.zeros_like(filtered)
    for n in range(4, len(filtered)):
        derivative[n] = (2 * filtered[n] + filtered[n - 1] - filtered[n - 3] - 2 * filtered[n - 4]) / 8.0

    # 3. Squaring function
    squared = derivative ** 2

    # 4. Moving Window Integration (MWI window ~ 150 ms)
    window_width = int(0.150 * fs)
    kernel = np.ones(window_width) / window_width
    integrated = np.convolve(squared, kernel, mode='same')

    # 5. Dual-Threshold Peak Detection
    # Signal and Noise running estimates
    peak_indices = []
    threshold_i1 = 0.5 * np.max(integrated[:fs * 2])
    spki = 0.125 * np.max(integrated[:fs * 2])
    npki = 0.125 * np.mean(integrated[:fs * 2])

    refractory_period = int(0.200 * fs) # 200 ms physiological refractory limit
    last_peak = -refractory_period

    for i in range(1, len(integrated) - 1):
        if integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1]:
            if integrated[i] > threshold_i1 and (i - last_peak) > refractory_period:
                # Search local window in original filtered signal for the sharp R-peak
                search_radius = int(0.100 * fs)
                start_w = max(0, i - search_radius)
                end_w = min(len(signal), i + search_radius)
                r_peak = start_w + np.argmax(np.abs(signal[start_w:end_w]))
                
                peak_indices.append(r_peak)
                last_peak = i
                spki = 0.125 * integrated[i] + 0.875 * spki
            else:
                npki = 0.125 * integrated[i] + 0.875 * npki

            threshold_i1 = npki + 0.25 * (spki - npki)

    return np.array(peak_indices, dtype=int)

if __name__ == '__main__':
    import wfdb
    # Test on the first 10 seconds of patient 100
    rec = wfdb.rdrecord('mitdb/100', sampto=3600, channels=[0])
    raw_sig = rec.p_signal[:, 0]
    detected_peaks = pan_tompkins_qrs(raw_sig, fs=360)
    print(f"Pan-Tompkins detected {len(detected_peaks)} R-peaks autonomously in 10 seconds.")