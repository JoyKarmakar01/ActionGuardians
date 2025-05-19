import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.actionguardian.utils.common import save_json
from src.actionguardian import logger
from dotenv import load_dotenv
from dvc.api import DVCFileSystem
from src.actionguardian.entity.config_entity import ModelEvaluationConfig

# Load environment variables
load_dotenv()

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, params: dict):
        self.cfg = config
        self.params = params

        # Set MLflow tracking URI from environment variable (securely loaded from .env)
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', str(config.mlflow_uri))
        mlflow.set_tracking_uri(mlflow_uri)

        # MLflow credentials
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        #Set tracking URI for local viewing
        # mlflow.set_tracking_uri("file:///C:/ActionGuardian/mlruns")


    def eval_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        out = os.path.join(self.cfg.root_dir, 'confusion_matrix.png')
        plt.savefig(out)
        plt.close()
        logger.info(f"Saved confusion matrix to {out}")

    def run(self):
        # Load test data
        X_test = np.load(self.cfg.test_features_path)
        y_test = np.load(self.cfg.test_labels_path)
        logger.info(f"Loaded test arrays: {self.cfg.test_features_path}, {self.cfg.test_labels_path}")

        # Load model and label encoder
        model = joblib.load(self.cfg.model_path)
        le = joblib.load(self.cfg.model_path.parent / 'label_encoder.pkl')
        class_names = list(le.classes_)

        # Prediction and metrics calculation
        y_pred = model.predict(X_test)
        metrics = self.eval_metrics(y_test, y_pred)
        save_json(self.cfg.metric_file_name, metrics)
        logger.info(f"Metrics saved to {self.cfg.metric_file_name}")

        # Plot and save confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, class_names)

        # Log metrics, parameters, and artifacts to MLflow
        scheme = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            # Log parameters
            for k, v in self.params.items():
                mlflow.log_param(k, v)
            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            # Log model
            if scheme != 'file':
                mlflow.sklearn.log_model(model, 'model', registered_model_name='ActivityRF')
            else:
                mlflow.sklearn.log_model(model, 'model')
            # Log artifacts
            mlflow.log_artifact(os.path.join(self.cfg.root_dir, 'confusion_matrix.png'))

        logger.info("Model evaluation complete & logged to MLflow")

        # DVC tracking - explicitly add artifacts directory
        os.system(f"dvc add {self.cfg.root_dir}")
        os.system("git add .")
        os.system('git commit -m "Track model evaluation artifacts with DVC"')
        os.system("dvc push")
        os.system("git push")
        logger.info("Artifacts tracked with DVC")
