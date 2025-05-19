import pandas as pd
import numpy as np
from scipy.fft import fft
from collections import Counter

def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def extract_features_from_windows(X_windows, sampling_rate=50):
    feature_list = []
    for window in X_windows:
        stats = []
        for i in range(window.shape[1]):
            col = window[:, i]
            stats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.sum(col ** 2)
            ])
            jerk = np.diff(col) * sampling_rate
            stats.extend([
                np.mean(jerk), np.std(jerk), np.sum(jerk ** 2)
            ])
            stats.append(zero_crossing_rate(col))
            fft_vals = np.abs(fft(col))
            stats.extend([
                np.mean(fft_vals), np.max(fft_vals), np.std(fft_vals)
            ])
        feature_list.append(stats)
    if not feature_list:
        raise ValueError("No features extracted. Possibly not enough data.")
    return np.array(feature_list)

def create_sliding_windows(df, window_size=250, step_size=250):
    feature_data = df.values
    timestamps = df.index.to_numpy()
    X, ts = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        X.append(feature_data[start:end])
        ts.append(timestamps[start])
    if not X:
        raise ValueError("No sliding windows could be created. Not enough data.")
    return np.array(X), np.array(ts)

def load_and_merge_sensor_data(acc_df, gyro_df):
    acc_sorted = acc_df.sort_values('timestamp')
    gyro_sorted = gyro_df.sort_values('timestamp')
    merged = pd.merge_asof(acc_sorted, gyro_sorted, on='timestamp', direction='nearest').dropna()
    merged = merged[merged['timestamp'] > merged['timestamp'].min() + 5.0].reset_index(drop=True)
    merged = merged.rename(columns={
        'x_x': 'acc_x', 'y_x': 'acc_y', 'z_x': 'acc_z',
        'x_y': 'gyro_x', 'y_y': 'gyro_y', 'z_y': 'gyro_z'
    })
    final_columns = ['acc_z', 'acc_y', 'acc_x', 'gyro_z', 'gyro_y', 'gyro_x']
    return merged[final_columns]

def summarize_activity_predictions(predictions, window_duration=5):
    counts = Counter(predictions)
    return {activity: count * window_duration for activity, count in counts.items()}

