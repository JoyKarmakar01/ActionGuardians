import os
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.actionguardian import logger
from src.actionguardian.utils.common import save_json

import pandas as pd


from src.actionguardian.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.cfg = config
        os.makedirs(self.cfg.root_dir, exist_ok=True)

    def load_data(self):
        X = np.load(self.cfg.features_path)
        y = np.load(self.cfg.labels_path)
        logger.info(f"Loaded features {self.cfg.features_path} and labels {self.cfg.labels_path}")
        return X, y

    def scale_features(self, X: np.ndarray):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        path = self.cfg.root_dir / self.cfg.scaler_filename
        joblib.dump(scaler, path)
        logger.info(f"Saved scaler to {path}")
        return X_scaled

    def encode_labels(self, y: np.ndarray):
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        path = self.cfg.root_dir / self.cfg.label_encoder_filename
        joblib.dump(encoder, path)
        logger.info(f"Saved label encoder to {path}")
        return y_enc
    
    def _save_test_split(self, X_test: np.ndarray, y_test: np.ndarray):
        """Save test features and labels as NumPy .npy files."""
        x_path = self.cfg.root_dir / "X_test.npy"
        y_path = self.cfg.root_dir / "y_test.npy"
        np.save(x_path, X_test)
        np.save(y_path, y_test)
        logger.info(f"Saved test split to {x_path} and {y_path}")


    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state
        )

        # Save test split as .npy
        self._save_test_split(X_test, y_test)


        clf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        # save model
        model_path = self.cfg.root_dir / self.cfg.model_filename
        joblib.dump(clf, model_path)
        logger.info(f"Saved model to {model_path}")
        # save metrics
        metrics = {"test_accuracy": acc}
        metrics_path = self.cfg.root_dir / self.cfg.metrics_filename
        save_json(metrics_path, metrics)
        logger.info(f"Saved metrics to {metrics_path}")
        return acc