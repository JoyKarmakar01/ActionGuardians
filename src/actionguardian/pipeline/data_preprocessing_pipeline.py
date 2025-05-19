from src.actionguardian.config.configuration import ConfigurationManager
from src.actionguardian.components.data_preprocessing import DataPreprocessor
from src.actionguardian import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_preprocessing(self):

        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessor = DataPreprocessor(config=data_preprocessing_config)
        df = data_preprocessor.load_data()
        X, y = data_preprocessor.create_sliding_windows(df)
        features = data_preprocessor.extract_features(X)
        data_preprocessor.save_numpy(X=features, y=y)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.initiate_data_preprocessing()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


