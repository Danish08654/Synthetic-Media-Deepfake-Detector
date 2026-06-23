import sys
import os

# ---------------- FIX PATH (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

sys.path.append(PROJECT_ROOT)

# ---------------- IMPORTS ----------------
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import torch
import tempfile
import io

from backend.model.xception import load_model
from backend.model.fft_utils import extract_fft_features
from backend.utils.preprocessing import preprocess_image_bytes, extract_video_frames

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Synthetic Media & Deepfake Detector")

DEVICE = "cpu"
THRESHOLD = 0.5

# ---------------- MODEL PATH (FIXED) ----------------
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "model",
    "xception_deepfake.pth"
)

# Debug (optional but useful)
st.write("Model Path:", MODEL_PATH)
st.write("Exists:", os.path.exists(MODEL_PATH))

# ---------------- LOAD MODEL ----------------
cnn_model = load_model(MODEL_PATH, DEVICE)

# ---------------- CORE FUNCTION ----------------
def predict_image(img_bytes):
    tensor, img_array, _ = preprocess_image_bytes(img_bytes)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        cnn_score = float(cnn_model(tensor).squeeze())

    fft_score, _ = extract_fft_features(img_array)
    fft_norm = min(fft_score * 2.5, 1.0)

    final_score = 0.7 * cnn_score + 0.3 * fft_norm
    prediction = "AI-Generated" if final_score >= THRESHOLD else "Real"

    return {
        "prediction": prediction,
        "fake_probability": final_score,
        "cnn_score": cnn_score,
        "fft_score": fft_norm
    }

# ---------------- UI ----------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image", "🎬 Video", "ℹ️ Info"])

# ---------------- IMAGE TAB ----------------
with tab1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])

    if uploaded:
        st.image(uploaded, use_container_width=True)

        if st.button("Analyze Image"):
            result = predict_image(uploaded.getvalue())

            st.success(result["prediction"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Fake Probability", f"{result['fake_probability']:.3f}")
            c2.metric("CNN Score", f"{result['cnn_score']:.3f}")
            c3.metric("FFT Score", f"{result['fft_score']:.3f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["fake_probability"] * 100,
                title={"text": "Fake %"},
                gauge={"axis": {"range": [0, 100]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

# ---------------- VIDEO TAB ----------------
with tab2:
    st.subheader("Upload Video")
    video = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])

    if video:
        if st.button("Analyze Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())
            path = tfile.name

            frames = extract_video_frames(path, max_frames=10)

            scores = []

            for frame in frames:
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="PNG")

                result = predict_image(buf.getvalue())
                scores.append(result["fake_probability"])

            avg_score = np.mean(scores)
            prediction = "AI-Generated" if avg_score >= THRESHOLD else "Real"

            st.success(prediction)
            st.line_chart(scores)

# ---------------- INFO TAB ----------------
with tab3:
    st.markdown("""
    ### About it's

    - CNN (XceptionNet) → visual artifacts detection  
    - FFT → frequency noise detection  
    - Fusion → 70% CNN + 30% FFT  

    """)
