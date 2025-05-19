from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import os
import joblib

from src.actionguardian.pipeline.prediction_pipeline import(
    load_and_merge_sensor_data,
    create_sliding_windows,
    extract_features_from_windows,
    summarize_activity_predictions
)

app = FastAPI(title="Activity-Summary API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load("artifacts/model_trainer/activity_model.pkl")

@app.post("/predict/", summary="Upload 2 CSV to get Activity Summary")
async def predict_activity(
    acc_file: UploadFile = File(..., description="Accelerometer CSV"),
    gyro_file: UploadFile = File(..., description="Gyroscope CSV")
):

    acc_path = UPLOAD_DIR / acc_file.filename
    gyro_path = UPLOAD_DIR / gyro_file.filename
    try:
        with open(acc_path, "wb") as f:
            f.write(await acc_file.read())
        with open(gyro_path, "wb") as f:
            f.write(await gyro_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded files: {e}")

    
    df = load_and_merge_sensor_data(str(acc_path), str(gyro_path))
    if df.empty:
        raise HTTPException(status_code=400, detail="Sensor data merge returned no rows. Check your timestamps/ranges.")

    X_windows, timestamps = create_sliding_windows(df)
    if X_windows.size == 0:
        raise HTTPException(status_code=400, detail="Not enough rows to form any window. Try more data or smaller window_size.")

    features = extract_features_from_windows(X_windows)
    if features.ndim != 2 or features.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Feature extraction yielded no valid features.")

    try:
        preds = model.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    summary = summarize_activity_predictions(preds, timestamps)
    plain_summary = {str(int(k)): int(v) for k, v in summary.items()}
    print(f"Prediction Summary: {plain_summary}")
    return {"activity_summary_seconds": plain_summary}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


