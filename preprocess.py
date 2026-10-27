# preprocess.py
import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from config import (
    DS1_TRAIN, DS2_TEST, AAMI_MAPPING, CLASS_TO_IDX,
    FS, WINDOW_SIZE, LOWCUT, HIGHCUT, ORDER
)

def butter_bandpass():
    nyq = 0.5 * FS
    b, a = butter(ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype='band')
    return b, a

def filter_signal(data, b, a):
    return filtfilt(b, a, data)

def map_symbol_to_idx(symbol):
    for group, symbols in AAMI_MAPPING.items():
        if symbol in symbols:
            return CLASS_TO_IDX[group]
    return None

def process_cohort(record_ids, data_dir='mitdb'):
    b, a = butter_bandpass()
    half_w = WINDOW_SIZE // 2
    beats, labels = [], []

    for rec in record_ids:
        path = os.path.join(data_dir, rec)
        record = wfdb.rdrecord(path, channels=[0])
        ann = wfdb.rdann(path, 'atr')
        signal = record.p_signal[:, 0]
        
        if np.isnan(signal).any():
            nan_mask = np.isnan(signal)
            signal[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), signal[~nan_mask])
            
        filtered = filter_signal(signal, b, a)
        
        for r_peak, sym in zip(ann.sample, ann.symbol):
            target = map_symbol_to_idx(sym)
            if target is None:
                continue
                
            start = r_peak - half_w
            end = r_peak + half_w
            
            if start >= 0 and end < len(filtered):
                beat = filtered[start:end]
                std = np.std(beat)
                beat = (beat - np.mean(beat)) / (std if std > 1e-6 else 1e-6)
                beats.append(beat)
                labels.append(target)
                
    X = np.expand_dims(np.array(beats, dtype=np.float32), axis=1)
    y = np.array(labels, dtype=np.int64)
    return X, y

if __name__ == '__main__':
    print("Processing DS1 (Training)...")
    X_train, y_train = process_cohort(DS1_TRAIN)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)

    print("Processing DS2 (Testing)...")
    X_test, y_test = process_cohort(DS2_TEST)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print("Preprocessing completed.")