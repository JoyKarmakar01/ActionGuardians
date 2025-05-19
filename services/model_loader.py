# services/model_loader.py
import joblib

def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
