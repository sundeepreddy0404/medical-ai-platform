# 🩺 Advanced AI Medical Intelligence Platform

An end-to-end medical image diagnostic platform built to process radiological images, perform automated disease pattern recognition using Deep Learning, explain predictions visually via Explainable AI (Grad-CAM), generate structured medical summaries using LLMs, and track evaluation records in a persistent relational database.

---

## 🔗 Live Links & Deliverables

- **Live Application URL:** [Streamlit Cloud Deployment](https://medical-ai-platform-rtrghfmryhlxnyfdpncdwz.streamlit.app/)
- **GitHub Repository:** [sundeepreddy0404/medical-ai-platform](https://github.com/sundeepreddy0404/medical-ai-platform)

---

## 🏛️ System Architecture

The platform architecture follows a modular decoupled data flow:

```
                +----------------------------------+
                |  User Uploads Radiological Scan   |
                +----------------------------------+
                                |
                                v
                +----------------------------------+
                |    Streamlit Web Interface UI     |
                +----------------------------------+
                                |
                                v
                +----------------------------------+
                | PyTorch ResNet50 Inference Engine |
                +----------------------------------+
                   /                            \
                  /                              \
                 v                                v
+--------------------------------+  +--------------------------------+
|   Explainable AI (Grad-CAM)    |  |    LLM Diagnostic Generator     |
|  Activations Saliency Overlay  |  |  OpenAI API / Fallback Engine   |
+--------------------------------+  +--------------------------------+
                 \                                /
                  \                              /
                   v                            v
                +----------------------------------+
                |    SQLite Persistent Audit Log    |
                +----------------------------------+
```

---

## ✨ Key Features & Technical Specifications

**Deep Learning Computer Vision Backbone**
- Uses a pre-trained ResNet50 Convolutional Neural Network backbone optimized with ImageNet weights.
- Normalizes inputs (224×224 matrix transformations) to extract structural spatial patterns across medical scans.

**Explainable AI (XAI) with Grad-CAM**
- Captures gradients from the target feature extraction layer (`layer4[-1]`).
- Computes activation-weighted feature maps to highlight high-density pixel saliency regions influencing predictions.

**LLM-Powered Medical Reporting**
- Connects to OpenAI `gpt-4o-mini` for automated generation of structured radiology summaries.
- Features a deterministic heuristic fallback engine to guarantee report generation even when API connectivity is limited.

**Database Persistence & Audit Logging**
- Powered by an embedded SQLite engine storing detailed inference logs: timestamp, filename, prediction label, confidence metric, and generated report text.

---

## 🛠️ Tech Stack & Dependencies

| Layer | Component | Technology |
|---|---|---|
| Frontend & UI | Presentation Layer | Streamlit |
| API Layer | REST API Framework | FastAPI, Uvicorn |
| Deep Learning | Neural Network Framework | PyTorch, Torchvision |
| Explainable AI | Heatmap Visualisations | Custom Grad-CAM Engine, OpenCV, PIL |
| Data Processing | Scientific Computing | NumPy, Pandas |
| LLM Integration | Generative Language Engine | OpenAI Python SDK |
| Database | Persistence Layer | SQLite3 |
| Environment | Containerization & Runtime | Python 3.11, Docker, Streamlit Cloud |

---

## 📂 Repository Layout

```
medical-ai-platform/
│
├── core.py              # Shared logic: model loading, Grad-CAM, LLM reporting, DB access
├── app.py               # Streamlit UI Entrypoint (imports core.py)
├── api.py                # FastAPI REST API Entrypoint (imports core.py)
├── requirements.txt     # Project Dependency List
├── Dockerfile             # Container Execution Blueprint
└── README.md              # Comprehensive Technical Documentation
```

`core.py` centralizes the model, Grad-CAM, LLM report generation, and database logic so both the Streamlit UI and the REST API run the exact same inference pipeline — no duplicated logic.

---

## 🔌 REST API

The platform exposes a REST API (FastAPI) for programmatic access to the diagnostic pipeline, independent of the Streamlit UI.

### Run the API locally
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness/readiness check |
| `POST` | `/predict` | Upload an image (JPG/PNG); returns prediction, confidence, Grad-CAM heatmap (base64 PNG), and LLM report. Also logs the result to the database. |
| `GET` | `/records?limit=100` | Returns recent prediction audit log entries, newest first |

### Example: `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_scan.jpg"
```

**Response**
```json
{
  "filename": "sample_scan.jpg",
  "prediction": "Normal / Unremarkable",
  "confidence": 0.8421,
  "llm_report": "**Automated Diagnostic Findings:** ...",
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Example: `/records`

```bash
curl "http://localhost:8000/records?limit=10"
```

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python: Version 3.10 or 3.11
- Git installed on your local machine

### Step-by-Step Installation

**1. Clone the Repository**
```bash
git clone https://github.com/sundeepreddy0404/medical-ai-platform.git
cd medical-ai-platform
```

**2. Create and Activate Virtual Environment**

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```dos
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Environment Variables Configuration (Optional)**

Set your OpenAI API key for live GPT report generation:

Linux/macOS:
```bash
export OPENAI_API_KEY="your-actual-api-key"
```

Windows:
```dos
set OPENAI_API_KEY="your-actual-api-key"
```

**5. Run the Application**

Streamlit UI:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your web browser.

REST API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

---

## 🐳 Docker Deployment Guide

To containerize and run the platform locally using Docker:

**Build the Container Image**
```bash
docker build -t medical-ai-platform .
```

**Execute the Docker Container**
```bash
docker run -p 8501:8501 medical-ai-platform
```

Navigate to [http://localhost:8501](http://localhost:8501) to view the active application instance.

---

## 📊 Database Schema Details

The SQLite table structure (`medical_records.db`) maintains the following schema:

```sql
CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    llm_report TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

---

## 📋 Evaluation Criteria Mapping

| Evaluation Standard | Technical Implementation Summary |
|---|---|
| Deep Learning Performance | Leverages PyTorch ResNet50 backbone with standard tensor transformations and ImageNet weight initialization. |
| Explainable AI (Grad-CAM) | Computes target convolutional gradient hooks to generate dynamic heatmaps without external black-box wrappers. |
| LLM Integration | Structured clinical prompt construction targeting GPT models with automatic offline fallback logic. |
| API Development | FastAPI REST service (`/predict`, `/records`, `/health`) sharing a single core inference pipeline with the Streamlit UI. |
| Database Design | Embedded SQLite schema tracking system queries, predictions, confidence percentages, and timestamps. |
| Software Engineering Best Practices | Clean single-file modular design, resource caching using `@st.cache_resource`, robust exception handling, and automated cloud deployments. |

---

## 👤 Project Information

- **Candidate:** Sundeep Reddy
- **Assignment Title:** Advanced AI Medical Intelligence Platform
- **Submission Evaluation For:** SN Matrix Software Pvt. Ltd.
