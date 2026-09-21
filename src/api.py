import logging
import time

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


MODEL_FILE = "models/equipment_failure_model.joblib"

MODEL_VERSION = "1.0.0"
API_VERSION = "1.0.0"

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "machine_age",
    "error_count",
]


# -------------------------------------------------
# LOGGING
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("equipment-api")


# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

model = joblib.load(MODEL_FILE)

logger.info(
    "Loaded model version %s from %s",
    MODEL_VERSION,
    MODEL_FILE,
)


# -------------------------------------------------
# CREATE API
# -------------------------------------------------

app = FastAPI(
    title="Equipment Failure Prediction API",
    version=API_VERSION,
)


# -------------------------------------------------
# SIMPLE MONITORING COUNTERS
# -------------------------------------------------

REQUEST_COUNT = 0
PREDICTION_COUNT = 0
ERROR_COUNT = 0


# -------------------------------------------------
# REQUEST MONITORING
# -------------------------------------------------

@app.middleware("http")
async def monitor_requests(
    request: Request,
    call_next,
):

    global REQUEST_COUNT
    global ERROR_COUNT

    REQUEST_COUNT += 1

    start_time = time.perf_counter()

    try:

        response = await call_next(request)

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if response.status_code >= 400:
            ERROR_COUNT += 1

        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response

    except Exception:

        ERROR_COUNT += 1

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Unhandled error method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )

        raise


# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------

class EquipmentInput(BaseModel):

    temperature: float = Field(
        ge=50,
        le=100,
    )

    vibration: float = Field(
        ge=0.5,
        le=5.0,
    )

    machine_age: int = Field(
        ge=1,
        le=15,
    )

    error_count: int = Field(
        ge=0,
        le=10,
    )


# -------------------------------------------------
# RESPONSE SCHEMA
# -------------------------------------------------

class PredictionResponse(BaseModel):

    failure: int
    model_version: str


# -------------------------------------------------
# HEALTH ENDPOINT
# -------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
    }


# -------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_equipment_failure(
    equipment: EquipmentInput,
):

    global PREDICTION_COUNT

    input_data = pd.DataFrame(
        [equipment.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(
        input_data
    )[0]

    PREDICTION_COUNT += 1

    logger.info(
        "prediction=%s model_version=%s",
        prediction,
        MODEL_VERSION,
    )

    return {
        "failure": int(prediction),
        "model_version": MODEL_VERSION,
    }


# -------------------------------------------------
# BASIC MONITORING ENDPOINT
# -------------------------------------------------

@app.get("/metrics")
def metrics():

    return {
        "requests_total": REQUEST_COUNT,
        "successful_predictions_total": PREDICTION_COUNT,
        "errors_total": ERROR_COUNT,
        "model_version": MODEL_VERSION,
    }