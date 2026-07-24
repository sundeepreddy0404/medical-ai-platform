# 🩺 Advanced AI Medical Intelligence Platform

An end-to-end medical image diagnostic platform that integrates Deep Learning, Explainable AI (Grad-CAM), Automated LLM Medical Reporting, REST APIs, and Persistent Database Audit Logging into a unified web application.

---

## 🔗 Live Application & Links

* **Live Web Application:** [Streamlit Cloud Deployment](https://medical-ai-platform-rtrghfmryhlxnyfdpncdwz.streamlit.app/)
* **GitHub Repository:** [sundeepreddy0404/medical-ai-platform](https://github.com/sundeepreddy0404/medical-ai-platform)

---

## 🏗️ System Architecture

```text
[ User Interface ]  --->  [ Streamlit Front-End ]
                                   │
                                   ▼
                   [ Deep Learning Inference Engine ]
                   ├─ PyTorch ResNet50 Classifier
                   └─ Grad-CAM Saliency Map Generator
                                   │
                                   ├────────────────────────┐
                                   ▼                        ▼
                        [ LLM Report Generator ]   [ Database Engine ]
                        └─ OpenAI / Heuristic      └─ SQLite Audit Log
