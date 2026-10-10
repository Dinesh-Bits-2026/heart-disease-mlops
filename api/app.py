"""FastAPI service for heart-disease risk prediction.

Endpoints
    GET  /health      liveness/readiness check (also confirms the model loads)
    POST /predict     JSON patient record in -> prediction + confidence out
    GET  /metrics     Prometheus metrics (request counts, latency, predictions)
    GET  /model-info  metadata about the packaged model
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import predict as predictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("heart-api")

METADATA_PATH = Path("models/model_metadata.json")
KNOWN_PATHS = {"/health", "/predict", "/metrics", "/model-info", "/docs", "/openapi.json"}

# ---- Prometheus metrics -------------------------------------------------------
REQUESTS = Counter("http_requests_total", "Total HTTP requests",
                   ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds",
                    ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["outcome"])
CONFIDENCE = Histogram("prediction_confidence", "Confidence of served predictions",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# ---- Request / response schemas ----------------------------------------------
class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 54, "sex": 1, "cp": 3, "trestbps": 130, "chol": 246, "fbs": 0,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
        "ca": 0, "thal": 3}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy")
    thal: float = Field(..., description="Thalassemia: 3 normal, 6 fixed, 7 reversible")

    @field_validator("thal")
    @classmethod
    def thal_must_be_known(cls, value: float) -> float:
        if value not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    risk_probability: float = Field(..., description="Probability of heart disease")


# ---- App ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        predictor.load_model()
        logger.info("Model loaded at startup")
    except Exception as exc:                      # app still starts; /health will report it
        logger.error("Model could not be loaded at startup: %s", exc)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path if request.url.path in KNOWN_PATHS else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
                    status, elapsed * 1000)


@app.get("/health")
def health():
    try:
        predictor.load_model()
    except Exception:
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(record: PatientRecord):
    try:
        result = predictor.predict([record.model_dump()])[0]
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    label = int(result["prediction"])
    confidence = float(result["confidence"])
    risk = confidence if label == 1 else 1 - confidence
    outcome = "disease" if label == 1 else "no_disease"

    PREDICTIONS.labels(outcome).inc()
    CONFIDENCE.observe(confidence)
    # Log the outcome only - never the patient's health data
    logger.info("prediction=%s confidence=%.3f", outcome, confidence)

    return PredictionResponse(prediction=label,
                              label="disease" if label == 1 else "no disease",
                              confidence=round(confidence, 4),
                              risk_probability=round(risk, 4))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return json.loads(METADATA_PATH.read_text())
