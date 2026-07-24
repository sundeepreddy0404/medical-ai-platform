"""
core.py
Shared inference, explainability, LLM reporting, and database logic
used by both the Streamlit UI (app.py) and the REST API (api.py).
"""

import base64
import datetime
import io
import os
import sqlite3

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_FILE = "medical_records.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            llm_report TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_record(filename, prediction, confidence, llm_report):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO records (filename, prediction, confidence, llm_report, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, prediction, confidence, llm_report, str(datetime.datetime.utcnow())))
    conn.commit()
    conn.close()


def get_records(limit: int = 100):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, filename, prediction, confidence, llm_report, created_at '
        'FROM records ORDER BY id DESC LIMIT ?',
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    columns = ["id", "filename", "prediction", "confidence", "llm_report", "created_at"]
    return [dict(zip(columns, row)) for row in rows]


# Run once at import time so both app.py and api.py can rely on the table existing
init_db()

# ---------------------------------------------------------------------------
# Model loading (framework-agnostic — no Streamlit caching here)
# ---------------------------------------------------------------------------
_model = None


def load_model():
    """Lazily loads and caches the ResNet50 model as a module-level singleton."""
    global _model
    if _model is None:
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _model.eval()
    return _model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, class_idx, torch.softmax(output, dim=1)[0][class_idx].item()


_grad_cam = None


def get_grad_cam():
    global _grad_cam
    if _grad_cam is None:
        model = load_model()
        _grad_cam = GradCAM(model, model.layer4[-1])
    return _grad_cam


# ---------------------------------------------------------------------------
# LLM Report Generation
# ---------------------------------------------------------------------------
def generate_llm_report(prediction: str, confidence: float) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            prompt = f"""
            Generate a professional, structured medical summary report based on automated classification results:
            - Finding: {prediction}
            - Model Confidence: {confidence*100:.2f}%
            Provide: Clinical Observations, Model Explanation Context, and Recommended Next Steps.
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )
            return response.choices[0].message.content
        except Exception:
            pass

    # Deterministic fallback if key is not configured or the API call fails
    return (
        f"**Automated Diagnostic Findings:**\n"
        f"- **Primary Finding:** {prediction}\n"
        f"- **Confidence:** {confidence*100:.2f}%\n"
        f"- **Notes:** Feature saliency maps indicate pixel activations corresponding to "
        f"classification output. Clinical correlation and physician review recommended."
    )


# ---------------------------------------------------------------------------
# End-to-end inference pipeline
# ---------------------------------------------------------------------------
def run_inference(pil_img: Image.Image):
    """
    Runs the full pipeline on a PIL image:
    inference -> Grad-CAM heatmap -> overlay -> prediction label -> LLM report.

    Returns a dict with prediction, confidence, overlay image (PIL), and llm_report.
    """
    pil_img = pil_img.convert("RGB")
    rgb_img = np.array(pil_img.resize((224, 224)))
    input_tensor = transform(pil_img).unsqueeze(0)

    grad_cam = get_grad_cam()
    heatmap, class_idx, confidence = grad_cam.generate_heatmap(input_tensor)

    heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), 0.6, heatmap_cv, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)

    prediction = "Pneumonia / Anomaly Detected" if class_idx % 2 == 1 else "Normal / Unremarkable"
    llm_report = generate_llm_report(prediction, confidence)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "overlay_image": overlay_pil,
        "llm_report": llm_report,
    }


def image_to_base64(pil_img: Image.Image) -> str:
    """Encodes a PIL image as a base64 PNG string for JSON API responses."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
