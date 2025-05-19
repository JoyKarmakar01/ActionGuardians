from src.actionguardian.constants import *
from src.actionguardian.utils.common import read_yaml, create_directories

from src.actionguardian.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    DataPreprocessingConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
   # SuggestionGeneratorConfig,
    #PredictionPipelineConfig,
)

class ConfigurationManager:
    def __init__(self,
                config_filepath=CONFIG_FILE_PATH,
                params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema  # Whole schema including ACCELEROMETER_COLUMNS, etc.

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir, config.output_data_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            base_data_dir=Path(config.base_data_dir),
            output_data_dir=Path(config.output_data_dir)
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Reads data_preprocessing section from config and returns a DataPreprocessingConfig object.
        """
        cfg = self.config.data_preprocessing
        # create root directory for preprocessing
        create_directories([cfg.root_dir])

        return DataPreprocessingConfig(
            root_dir=Path(cfg.root_dir),
            data_path=Path(cfg.data_path),
            window_size=cfg.window_size,
            step_size=cfg.step_size,
            sampling_rate=cfg.sampling_rate
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config.model_trainer
        params = self.params.RandomForestClassifier  # ← match your YAML key

        create_directories([cfg.root_dir])

        return ModelTrainerConfig(
            root_dir = Path(cfg.root_dir),
            scaler_filename = cfg.scaler_filename,
            label_encoder_filename = cfg.label_encoder_filename,
            model_filename = cfg.model_filename,
            metrics_filename = cfg.metrics_filename,
            test_size = cfg.test_size,
            random_state = params.random_state,
            n_estimators = params.n_estimators,
            # max_depth   = params.max_depth,      # optional
            features_path = Path(cfg.features_path),
            labels_path   = Path(cfg.labels_path),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config.model_evaluation
        # ensure output directory exists
        create_directories([cfg.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            test_features_path=Path(cfg.test_features_path),
            test_labels_path=Path(cfg.test_labels_path),
            model_path=Path(cfg.model_path),
            params_section=cfg.params_section,
            metric_file_name=Path(cfg.metric_file_name),
            mlflow_uri=cfg.mlflow_uri
        )
