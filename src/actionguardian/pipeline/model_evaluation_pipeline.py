from src.actionguardian.config.configuration import ConfigurationManager
from src.actionguardian.components.model_evaluation import ModelEvaluation
from src.actionguardian import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        manager = ConfigurationManager()
        evaluation_config = manager.get_model_evaluation_config()
        all_params = manager.params.get(evaluation_config.params_section, {})
        evaluator = ModelEvaluation(config = evaluation_config,params=all_params)
        evaluator.run()
        print(" Model evaluation completed.")



# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = ModelEvaluationTrainingPipeline()
#         obj.initiate_model_evaluation()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<")
#     except Exception as e:
#         logger.exception(e)
#         raise e

        