import sys

from src.actionguardian import logger
from src.actionguardian.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.actionguardian.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.actionguardian.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.actionguardian.pipeline.data_preprocessing_pipeline import DataPreprocessingTrainingPipeline
from src.actionguardian.pipeline.model_training_pipeline import ModelTrainingTrainingPipeline
from src.actionguardian.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline


def run_stage(name: str, pipeline_cls: type, method: str) -> None:
    """
    Execute a pipeline stage with standardized logging and error handling.
    """
    logger.info(f"Starting {name}...")
    try:
        pipeline = pipeline_cls()
        getattr(pipeline, method)()
        logger.info(f"Finished {name} successfully.")
    except Exception:
        logger.exception(f"Failed {name}.")
        sys.exit(1)


def main() -> None:
    
    stages = [
        ("Data Ingestion", DataIngestionTrainingPipeline, "initiate_data_ingestion"),
        ("Data Validation", DataValidationTrainingPipeline, "initiate_data_validation"),
        ("Data Transformation", DataTransformationTrainingPipeline, "initiate_data_transformation"),
        ("Data Preprocessing", DataPreprocessingTrainingPipeline, "initiate_data_preprocessing"),
        ("Model Training", ModelTrainingTrainingPipeline, "initiate_model_training"),
        ("Model Evaluation", ModelEvaluationTrainingPipeline, "initiate_model_evaluation"),
    ]

    for stage_name, cls, method in stages:
        run_stage(stage_name, cls, method)


if __name__ == "__main__":
    main()

