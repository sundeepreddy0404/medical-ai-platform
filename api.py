"""
api.py
REST API for the Advanced AI Medical Intelligence Platform.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from core import get_records, image_to_base64, load_model, run_inference, save_record

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform API",
    description="REST API for medical image classification, Grad-CAM explainability, "
                 "and LLM-generated diagnostic reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    llm_report: str
    heatmap_base64: str  # PNG image, base64-encoded


class RecordResponse(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    llm_report: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
def _warm_up_model():
    # Load the model once at startup rather than on the first request
    load_model()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Basic liveness/readiness check."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Diagnostics"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a medical scan (JPG/PNG) and receive:
    - predicted condition
    - model confidence
    - Grad-CAM heatmap overlay (base64 PNG)
    - LLM-generated diagnostic report

    The result is also persisted to the audit log database.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="File must be a JPG or PNG image.")

    raw_bytes = await file.read()
    try:
        pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    result = run_inference(pil_img)
    save_record(file.filename, result["prediction"], result["confidence"], result["llm_report"])

    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "llm_report": result["llm_report"],
        "heatmap_base64": image_to_base64(result["overlay_image"]),
    }


@app.get("/records", response_model=list[RecordResponse], tags=["Audit Log"])
def list_records(limit: int = 100):
    """Returns the most recent prediction audit log entries, newest first."""
    return get_records(limit=limit)


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Advanced AI Medical Intelligence Platform API",
        "docs": "/docs",
        "health": "/health",
    }
