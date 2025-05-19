from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# -----------------------------
# ✅ Data Ingestion Config
# -----------------------------
@dataclass
class DataIngestionConfig:
    root_dir : Path
    source_URL : str
    local_data_file : Path
    unzip_dir : Path

# -----------------------------
# ✅ Data Validation Config
# -----------------------------
@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: Dict

# -----------------------------
# ✅ Data Transformation Config
# -----------------------------
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    base_data_dir: Path
    output_data_dir: Path



# -----------------------------
# ✅ Data Preprocessing Config
# -----------------------------
@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path            # artifacts root for preprocessed outputs
    data_path: Path           # path to merged CSV (data.csv)
    window_size: int          # sliding window length (samples)
    step_size: int            # sliding window step (samples)
    sampling_rate: int        # for jerk calculation and FFT


# -----------------------------
# ✅ Model Training Config
# -----------------------------
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    scaler_filename: str
    label_encoder_filename: str
    model_filename: str
    metrics_filename: str
    test_size: float
    random_state: int
    n_estimators: int
    features_path: Path  # Path to precomputed features (.npy)
    labels_path: Path    # Path to precomputed labels (.npy)

# -----------------------------
# ✅ Model Evaluation Config
# -----------------------------
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_features_path: Path     # was: test_data_path
    test_labels_path: Path       # new
    model_path: Path
    params_section: str
    metric_file_name: Path
    mlflow_uri: str




# -----------------------------
# ✅ Suggestion Generator Config
# -----------------------------
@dataclass
class SuggestionGeneratorConfig:
    model_name: str
    json_summary_path: Path
    output_text_path: Path
    max_tokens: int = 100
    temperature: float = 0.7

# -----------------------------
# ✅ Prediction Pipeline Config
# -----------------------------
@dataclass
class PredictionPipelineConfig:
    model_path: Path
    scaler_path: Path
    label_encoder_path: Path
    input_data_path: Path
    output_summary_path: Path
    window_size: int
    stride: int
