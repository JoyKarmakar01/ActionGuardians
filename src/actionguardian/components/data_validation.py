import os
import pandas as pd
from pathlib import Path

from src.actionguardian.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config):
        self.config = config
        self.acc_schema = config.all_schema["ACCELEROMETER_COLUMNS"]
        self.gyro_schema = config.all_schema["GYROSCOPE_COLUMNS"]

    def validate_csv_file(self, csv_path, expected_columns):
        df = pd.read_csv(csv_path)
        actual_cols = list(df.columns)
        return set(actual_cols) == set(expected_columns)

    def validate_all_sensor_files(self) -> bool:
        base_dir = self.config.unzip_data_dir
        validation_status = True

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file in ["Accelerometer.csv", "Gyroscope.csv"]:
                    file_path = Path(root) / file
                    if "Accelerometer" in file:
                        is_valid = self.validate_csv_file(file_path, self.acc_schema.keys())
                    else:
                        is_valid = self.validate_csv_file(file_path, self.gyro_schema.keys())

                    if not is_valid:
                        validation_status = False
                        print(f"Invalid columns in: {file_path}")
                    else:
                        print(f"Validated: {file_path}")
        
        # Write status to file
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(f"Validation status: {validation_status}")
        
        return validation_status