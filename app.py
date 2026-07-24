import sqlite3

import pandas as pd
import streamlit as st
from PIL import Image

from core import DB_FILE, run_inference, save_record

# Page Config
st.set_page_config(page_title="Advanced AI Medical Intelligence Platform", layout="wide")

# Streamlit UI Layout
st.title("🩺 Advanced AI Medical Intelligence Platform")
st.markdown("### Deep Learning Diagnostics • Explainable AI (Grad-CAM) • Automated LLM Reports")

tab1, tab2 = st.tabs(["🔬 Image Diagnostic Pipeline", "📜 Database Audit Logs"])

with tab1:
    uploaded_file = st.file_uploader("Upload Medical Scan (Chest X-Ray / Scan)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        pil_img = Image.open(uploaded_file).convert('RGB')

        with col1:
            st.image(pil_img, caption="Original Medical Image", use_container_width=True)

        if st.button("Run AI Medical Analysis", type="primary"):
            with st.spinner("Executing deep learning inference & generating Grad-CAM heatmap..."):
                result = run_inference(pil_img)
                save_record(
                    uploaded_file.name,
                    result["prediction"],
                    result["confidence"],
                    result["llm_report"],
                )

                with col2:
                    st.image(result["overlay_image"], caption="Grad-CAM Visual Explanation", use_container_width=True)

                st.divider()
                st.subheader("📊 Diagnostic Summary")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Condition", result["prediction"])
                c2.metric("Model Confidence", f"{result['confidence']*100:.2f}%")

                st.subheader("📝 AI-Generated Medical Report")
                st.info(result["llm_report"])

with tab2:
    st.subheader("Prediction History & Audit Trail")
    if st.button("Refresh Audit Logs"):
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM records ORDER BY id DESC", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)
