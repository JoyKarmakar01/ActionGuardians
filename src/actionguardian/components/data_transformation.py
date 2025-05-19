import os
import pandas as pd
from glob import glob
from typing import List
from datetime import datetime, timedelta
from src.actionguardian import logger
from pathlib import Path


from src.actionguardian.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.base_dir = config.base_data_dir
        self.output_dir = config.output_data_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_merge(self, accel_path: str, gyro_path: str, duration: int = 60) -> pd.DataFrame:
        acc = pd.read_csv(accel_path)
        gyro = pd.read_csv(gyro_path)

        acc['seconds_elapsed'] = acc['seconds_elapsed'].round(3)
        gyro['seconds_elapsed'] = gyro['seconds_elapsed'].round(3)

        merged = pd.merge(acc, gyro, on='seconds_elapsed', suffixes=('_acc', '_gyro'))
        merged = merged[(merged['seconds_elapsed'] > 10) & (merged['seconds_elapsed'] <= 10 + duration)]
        merged.drop(columns=['time_acc', 'time_gyro'], inplace=True, errors='ignore')
        return merged

    def _process_activity(self, folder_paths: List[str], label: str, duration: int = 60) -> pd.DataFrame:
        dfs = []
        for folder in folder_paths:
            accel_path = os.path.join(folder, "Accelerometer.csv")
            gyro_path = os.path.join(folder, "Gyroscope.csv")

            if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
                logger.warning(f"Missing file in {folder}")
                continue

            df = self._load_and_merge(accel_path, gyro_path, duration)
            df['label'] = label
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def transform_and_save(self) -> pd.DataFrame:
        activity_map = {
            "upstair": glob(os.path.join(self.base_dir, "Upstair_all_1", "Upstair_*")),
            "downstair": glob(os.path.join(self.base_dir, "Downstair_all_1", "Downstairs_*")),
            "jogging": [os.path.join(self.base_dir, "Jogging_1")],
            "sitting": [os.path.join(self.base_dir, "Sitting_1")],
            "standing": [os.path.join(self.base_dir, "Standing_1")],
            "walking": [os.path.join(self.base_dir, "Walking_1")],
        }

        durations = {"upstair": 60, "downstair": 60, "jogging": 300, "sitting": 300, "standing": 300, "walking": 300}
        all_dfs = []
        file_map = {}

        for idx, (label, folders) in enumerate(activity_map.items(), start=1):
            df = self._process_activity(folders, label, duration=durations[label])
            if not df.empty:
                df['id'] = idx
                file_path = os.path.join(self.output_dir, f"{label}_final.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {label} to {file_path}")
                all_dfs.append(df)
                file_map[idx] = file_path

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df['timestamp'] = final_df['seconds_elapsed'].apply(lambda x: (datetime(2025, 1, 1) + timedelta(seconds=x)).strftime("%Y-%m-%d %H:%M:%S"))

        merged_csv = os.path.join(self.output_dir, "data.csv")
        final_df.to_csv(merged_csv, index=False)
        logger.info(f"Merged final dataset saved to {merged_csv}")

        return final_df