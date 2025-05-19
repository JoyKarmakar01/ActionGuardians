import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import fft
from collections import Counter

# ---------------------- Helper Functions ----------------------

def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def extract_features_from_windows(X_windows, sampling_rate=50):
    try:
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
        features = np.array(feature_list)
        if features.size == 0:
            raise ValueError("No features extracted. Possibly not enough data.")
        return features
    except Exception as e:
        print(f"Error in extract_features_from_windows: {e}")
        return np.empty((0,))

def create_sliding_windows(df, window_size=250, step_size=250):
    try:
        X = []
        timestamps = []
        df = df.drop(columns=['timestamp'], errors='ignore')
        feature_data = df.values
        all_timestamps = df.index.to_numpy()

        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size
            window = feature_data[start:end]
            ts_window = all_timestamps[start:end]
            if len(window) == window_size:
                X.append(window)
                timestamps.append(ts_window[0])
        if len(X) == 0:
            raise ValueError("No sliding windows could be created. Not enough data.")
        return np.array(X), np.array(timestamps)
    except Exception as e:
        print(f"Error in create_sliding_windows: {e}")
        return np.empty((0, 0, 0)), np.array([])

# ✅ Updated to enforce clean column names and correct order
def load_and_merge_sensor_data(acc_path, gyro_path):
    try:
        acc_df = pd.read_csv(acc_path)
        gyro_df = pd.read_csv(gyro_path)
        
        # 2) If your CSVs call the time-column something else, rename it:
        if "seconds_elapsed" in acc_df.columns:
            acc_df = acc_df.rename(columns={"seconds_elapsed": "timestamp"})
        if "seconds_elapsed" in gyro_df.columns:
            gyro_df = gyro_df.rename(columns={"seconds_elapsed": "timestamp"})

        merged = pd.merge_asof(
            acc_df.sort_values('timestamp'),
            gyro_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )

        merged = merged.dropna()
        merged = merged[merged['timestamp'] > merged['timestamp'].min() + 5.0]
        merged = merged.reset_index(drop=True)

        if merged.empty:
            raise ValueError("Merged dataframe is empty after cleaning.")

        # Rename columns for clarity
        merged = merged.rename(columns={
            'x_x': 'acc_x', 'y_x': 'acc_y', 'z_x': 'acc_z',
            'x_y': 'gyro_x', 'y_y': 'gyro_y', 'z_y': 'gyro_z'
        })

        # Keep only the necessary 6 columns in your required order
        final_columns = ['acc_z', 'acc_y', 'acc_x', 'gyro_z', 'gyro_y', 'gyro_x']
        merged = merged[final_columns]

        print(f"Merged shape: {merged.shape}")
        print(f"Columns used: {list(merged.columns)}")
        print(f"Sample:\n{merged.head()}")

        return merged
    except Exception as e:
        print(f"Error in load_and_merge_sensor_data: {e}")
        return pd.DataFrame()

def summarize_activity_predictions(predictions, timestamps, window_duration=5):
    activity_counts = Counter(predictions)
    summary = {activity: count * window_duration for activity, count in activity_counts.items()}
    return summary

# ---------------------- Main Pipeline ----------------------

def main():
    acc_path = 'test/Accelerometer.csv'
    gyro_path = 'test/Gyroscope.csv'
    model_path = 'artifacts/model_trainer/activity_model.pkl'


    df = load_and_merge_sensor_data(acc_path, gyro_path)
    if df.empty:
        print("Stopping: Sensor data couldn't be loaded properly.")
        return

    X_windows, timestamps = create_sliding_windows(df)
    if X_windows.size == 0:
        print("Stopping: No windows created from data.")
        return

    features = extract_features_from_windows(X_windows)
    if features.size == 0:
        print("Stopping: Feature extraction failed.")
        return

    try:
        model = joblib.load(model_path)
        predictions = model.predict(features)
    except Exception as e:
        print(f"Error loading or predicting with model: {e}")
        return

    summary = summarize_activity_predictions(predictions, timestamps)
    print("\n Activity Summary (Duration in Seconds):")
    for activity, duration in summary.items():
        print(f"{activity}: {duration} sec")
    return summary
if __name__ == '__main__':
    summary = main()
