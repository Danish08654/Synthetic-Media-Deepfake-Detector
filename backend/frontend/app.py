import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go

API_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Synthetic Media & Deepfake Detector")
st.caption("Hybrid CNN + FFT detection for images and video")

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image",
    "🎬 Video",
    "ℹ️ Model Info"
])


def render_verdict(result):
    is_fake = result['fake_probability'] >= 0.5
    color   = "#e74c3c" if is_fake else "#27ae60"
    label   = result['prediction']

    st.markdown(
        f"<div style='background:{color};padding:16px;border-radius:10px;"
        f"text-align:center;color:white;font-size:22px;font-weight:700'>"
        f"{label}</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction",      result['prediction'])
    c2.metric("Fake Probability", f"{result['fake_probability']*100:.1f}%")
    c3.metric("CNN Score",        f"{result.get('cnn_score', 0):.3f}")
    c4.metric("FFT Score",        f"{result.get('fft_score', 0):.3f}")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(result['fake_probability'] * 100, 1),
        number= {"suffix": "% Fake"},
        gauge = {
            "axis":  {"range": [0, 100]},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#d5f5e3"},
                {"range": [35, 65], "color": "#fef9e7"},
                {"range": [65,100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "value": 50
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 1: Image ---
with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Upload Image")
        uploaded = st.file_uploader(
            "Upload face or profile image",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded:
            st.image(uploaded, use_column_width=True)
            if st.button("Analyse Image", type="primary",
                         use_container_width=True):
                with st.spinner("Running hybrid detection..."):
                    try:
                        files  = {"image": (uploaded.name,
                                            uploaded.getvalue(),
                                            uploaded.type)}
                        resp   = requests.post(
                            f"{API_URL}/detect/image", files=files
                        )
                        result = resp.json()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        st.stop()

                with col_b:
                    st.subheader("Detection Result")
                    render_verdict(result)
        else:
            with col_b:
                st.info("Upload an image and click **Analyse Image** to begin.")


# --- Tab 2: Video ---
with tab2:
    st.subheader("Video Deepfake Detection")
    video_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        if st.button("Analyse Video", type="primary",
                     use_container_width=True):
            with st.spinner("Extracting and analysing frames..."):
                try:
                    files  = {"video": (video_file.name,
                                        video_file.getvalue(),
                                        video_file.type)}
                    resp   = requests.post(
                        f"{API_URL}/detect/video", files=files
                    )
                    result = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            render_verdict({
                'prediction':       result['overall_prediction'],
                'fake_probability': result['avg_fake_score'],
                'cnn_score':        result['avg_fake_score'],
                'fft_score':        0,
                'confidence':       result['confidence']
            })

            st.divider()
            st.subheader(f"Frame Analysis — {result['frames_analyzed']} frames")

            frame_scores = [f['fake_probability']
                            for f in result['frame_results']]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x    = list(range(1, len(frame_scores) + 1)),
                y    = frame_scores,
                mode = 'lines+markers',
                line = dict(color='tomato', width=2),
                name = 'Fake Score'
            ))
            fig.add_hline(
                y=0.5, line_dash="dash",
                line_color="gray",
                annotation_text="Threshold (0.5)"
            )
            fig.update_layout(
                xaxis_title="Frame",
                yaxis_title="Fake Probability",
                height=300,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Per-Frame Results")
            for f in result['frame_results']:
                icon = "🔴" if f['fake_probability'] >= 0.5 else "🟢"
                st.markdown(
                    f"{icon} Frame {f['frame']} — "
                    f"Score: {f['fake_probability']:.3f} — "
                    f"{f['prediction']}"
                )
    else:
        st.info("Upload a video file and click **Analyse Video** to begin.")


# --- Tab 3: Model Info ---
with tab3:
    st.subheader("Model Information")
    try:
        cfg = requests.get(f"{API_URL}/").json()

        c1, c2, c3 = st.columns(3)
        c1.metric("CNN Model",  "XceptionNet")
        c2.metric("Hybrid AUC", cfg.get('hybrid_auc', 'N/A'))
        c3.metric("Fusion",     "0.7×CNN + 0.3×FFT")

        st.divider()
        st.markdown("""
        ### How Detection Works

        **Spatial Detection — XceptionNet (70% weight)**
        Depthwise separable convolutions detect blending artifacts,
        unnatural textures, and face boundary inconsistencies left
        behind by GAN generation and face-swapping algorithms.

        **Frequency Detection — FFT (30% weight)**
        Fast Fourier Transform reveals GAN fingerprints invisible
        to the naked eye — periodic noise patterns and abnormal
        high-frequency energy distributions consistent across
        AI-generated images.

        **Hybrid Fusion Score**
        ```
        Final Score = 0.7 × CNN Score + 0.3 × FFT Score
        ```
        Score above 0.5 = AI-Generated.
        Score below 0.5 = Real.

        **Chrome Extension**
        The `/detect/url` endpoint powers the browser extension
        which scans images on any webpage in real time.
        Green = Real · Red = AI-Generated · Yellow = Uncertain.
        """)

    except Exception:
        st.error("Cannot connect to API at http://localhost:5000 — make sure Flask is running.")