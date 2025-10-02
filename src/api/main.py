from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
import joblib
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

app = FastAPI(title="Irrigation Linear Regression API", version="1.0.0")

class PredictRequest(BaseModel):
    irrigation_hours: float = Field(..., ge=0, description="Hours of irrigation")

class PredictBatchRequest(BaseModel):
    irrigation_hours: conlist(float, min_length=1)

class PredictResponse(BaseModel):
    irrigated_area_per_angle: float

class PredictBatchResponse(BaseModel):
    irrigated_area_per_angle: list[float]

@app.on_event("startup")
def load_artifacts():
    model_path = ARTIFACTS_DIR / "model.joblib"
    meta_path = ARTIFACTS_DIR / "metadata.json"
    if not model_path.exists():
        raise RuntimeError("Model artifact not found. Train the model first.")
    app.state.model = joblib.load(model_path)
    app.state.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in app.state.__dict__}

@app.get("/model-info")
def model_info():
    return app.state.metadata

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array([[req.irrigation_hours]], dtype=float)
        y_pred = app.state.model.predict(x).ravel()[0]
        return {"irrigated_area_per_angle": float(y_pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    try:
        X = np.array(req.irrigation_hours, dtype=float).reshape(-1, 1)
        y_pred = app.state.model.predict(X).ravel().tolist()
        return {"irrigated_area_per_angle": [float(v) for v in y_pred]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))