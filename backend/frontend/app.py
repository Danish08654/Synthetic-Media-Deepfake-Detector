import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Detector AI",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: white;
}
.real {
    background: linear-gradient(90deg, #00b09b, #96c93d);
}
.fake {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}
.card {
    padding: 15px;
    border-radius: 10px;
    background: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Deepfake Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid CNN + FFT Analysis | Streamlit Cloud Ready</div>', unsafe_allow_html=True)

# ---------------- MODEL ----------------
DEVICE = "cpu"
THRESHOLD = 0.5

MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "model", "xception_deepfake.pth")
cnn_model = load_model(MODEL_PATH, DEVICE)

# ---------------- PREDICTION ----------------
def predict_image(img_bytes):
    tensor, img_array, _ = preprocess_image_bytes(img_bytes)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        cnn_score = float(cnn_model(tensor).squeeze())

    fft_score, _ = extract_fft_features(img_array)
    fft_norm = min(fft_score * 2.5, 1.0)

    final_score = 0.7 * cnn_score + 0.3 * fft_norm
    prediction = "AI-Generated" if final_score >= THRESHOLD else "Real"

    return prediction, final_score, cnn_score, fft_norm

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎬 Video Analysis", "ℹ️ About Model"])

# ================= IMAGE TAB =================
with tab1:
    st.subheader("Upload Image for Analysis")

    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded, use_container_width=True)

        if st.button("🔍 Analyze Image"):
            prediction, score, cnn, fft = predict_image(uploaded.getvalue())

            is_fake = score >= THRESHOLD

            with col2:
                st.markdown(
                    f"""
                    <div class="result-box {'fake' if is_fake else 'real'}">
                        {prediction}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(score)

                c1, c2, c3 = st.columns(3)
                c1.metric("Fake Probability", f"{score:.2f}")
                c2.metric("CNN Score", f"{cnn:.2f}")
                c3.metric("FFT Score", f"{fft:.2f}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    title={"text": "AI Probability (%)"},
                    gauge={"axis": {"range": [0, 100]}}
                ))

                st.plotly_chart(fig, use_container_width=True)

# ================= VIDEO TAB =================
with tab2:
    st.subheader("Video Deepfake Detection")

    video = st.file_uploader("", type=["mp4", "avi", "mov"])

    if video:
        if st.button("🔍 Analyze Video"):

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())
            path = tfile.name

            frames = extract_video_frames(path, max_frames=10)

            scores = []

            for frame in frames:
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="PNG")

                _, score, _, _ = predict_image(buf.getvalue())
                scores.append(score)

            avg_score = np.mean(scores)
            prediction = "AI-Generated" if avg_score >= THRESHOLD else "Real"

            st.markdown(
                f"""
                <div class="result-box {'fake' if avg_score >= THRESHOLD else 'real'}">
                    {prediction}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(avg_score)

            st.subheader("Frame Analysis")
            st.line_chart(scores)

# ================= INFO TAB =================
with tab3:
    st.subheader("Model Architecture")

    st.markdown("""
    ### 🧠 How it works

    **1. CNN (XceptionNet - 70%)**
    - Detects facial blending artifacts  
    - Learns GAN-generated inconsistencies  

    **2. FFT Analysis (30%)**
    - Detects frequency-level noise patterns  
    - Finds GAN fingerprints invisible to humans  

    **3. Final Fusion**
    ```
    Final Score = 0.7 × CNN + 0.3 × FFT
    ```

    **Output**
    - Score ≥ 0.5 → AI Generated  
    - Score < 0.5 → Real
    """)

    st.success("Optimized for Streamlit Cloud 🚀")
