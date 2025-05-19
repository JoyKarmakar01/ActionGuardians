import os
import pandas as pd
import numpy as np
from scipy.fft import fft
from glob import glob
from typing import Tuple
from src.actionguardian import logger

from src.actionguardian.entity.config_entity import DataPreprocessingConfig

class DataPreprocessor:
    def __init__(self, config: DataPreprocessingConfig):
        self.cfg = config
        os.makedirs(self.cfg.root_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.data_path)
        logger.info(f"Loaded data for preprocessing: {self.cfg.data_path} (shape={df.shape})")
        return df

    def create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        df = df.drop(columns=['id','timestamp','seconds_elapsed'], errors='ignore')
        data_vals = df.drop(columns=['label']).values
        labels = df['label'].values

        for start in range(0, len(df) - self.cfg.window_size + 1, self.cfg.step_size):
            window = data_vals[start:start + self.cfg.window_size]
            label_mode = pd.Series(labels[start:start + self.cfg.window_size]).mode()[0]
            if window.shape[0] == self.cfg.window_size:
                X.append(window)
                y.append(label_mode)

        X_arr = np.stack(X)
        y_arr = np.array(y)
        logger.info(f"Created {len(X_arr)} windows of size {self.cfg.window_size}")
        return X_arr, y_arr

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        def zero_cross_rate(sig):
            return ((sig[:-1] * sig[1:]) < 0).sum()

        feats = []
        for window in X:
            stats = []
            num_features = window.shape[-1]  # support 2D windows
            for i in range(num_features):
                col = window[:, i]
                # Time-domain
                stats += [np.mean(col), np.std(col), np.min(col), np.max(col), np.median(col), np.sum(col**2)]
                # Jerk
                jerk = np.diff(col) * self.cfg.sampling_rate
                stats += [np.mean(jerk), np.std(jerk), np.sum(jerk**2)]
                # Zero crossing
                stats.append(zero_cross_rate(col))
                # Frequency-domain
                fft_vals = np.abs(fft(col))
                stats += [np.mean(fft_vals), np.max(fft_vals), np.std(fft_vals)]
            feats.append(stats)
        feat_arr = np.stack(feats)
        logger.info(f"Extracted features: {feat_arr.shape[1]} features per {feat_arr.shape[0]} windows")
        return feat_arr

    def save_numpy(self, X: np.ndarray, y: np.ndarray):
        np.save(os.path.join(self.cfg.root_dir, 'X_windows.npy'), X)
        np.save(os.path.join(self.cfg.root_dir, 'y_labels.npy'), y)
        logger.info(f"Saved sliding windows and labels to {self.cfg.root_dir}")