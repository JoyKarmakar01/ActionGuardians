from src.actionguardian.config.configuration import ConfigurationManager
from src.actionguardian.components.model_trainer import ModelTrainer
from src.actionguardian import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        X, y = model_trainer.load_data()
        X_scaled = model_trainer.scale_features(X)
        y_enc = model_trainer.encode_labels(y)
        accuracy = model_trainer.train_and_evaluate(X_scaled, y_enc)
        print(f"Model training completed. Test Accuracy: {accuracy:.4f}")




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingTrainingPipeline()
        obj.initiate_model_training()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
