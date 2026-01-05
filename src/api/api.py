# src/api/app.py

from fastapi import FastAPI, HTTPException
from src.api.models import Transaction
import mlflow
import numpy as np

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves fraud prediction model from MLflow Model Registry",
    version="1.0.0",
)

# =========================
# MLflow Model Config
# =========================

MODEL_NAME = "fraud_detection"
MODEL_ALIAS = "production"  # Use alias (MLflow >=2.x)

# MLflow URI using alias
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = None  # Global model object

# =========================
# Load Model on Startup
# =========================

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"📦 Loading model from MLflow Registry: {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        model = None

# =========================
# Health Check Endpoint
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }

# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input features to numpy array
    data = np.array(transaction.features).reshape(1, -1)

    # Validate feature dimension
    expected_features = 30  # Adjust according to processed dataset
    if data.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} features, got {data.shape[1]}"
        )

    try:
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "prediction": int(prediction[0]),
        "is_fraud": bool(prediction[0] == 1),
        "probability": float(probability[0]) if probability is not None else None
    }
