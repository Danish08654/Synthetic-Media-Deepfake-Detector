import json
import io
import os
import tempfile
import numpy as np
import torch
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from model.xception import XceptionDetector, load_model
from model.fft_utils import extract_fft_features
from utils.preprocessing import preprocess_image_bytes, extract_video_frames

app  = Flask(__name__)
CORS(app)

# --- Paths ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'deepfake_config.json')

# --- Load config ---
with open(CONFIG_PATH) as f:
    config = json.load(f)

THRESHOLD  = config['threshold']
CNN_WEIGHT = config['cnn_weight']
FFT_WEIGHT = config['fft_weight']
DEVICE     = 'cpu'

# --- Load XceptionNet ---
print("Loading XceptionNet...")
cnn_model = load_model(
    os.path.join(BASE_DIR, 'xception_deepfake.pth'), DEVICE
)
print("All models ready.")


def predict_image(img_bytes: bytes) -> dict:
    """Hybrid CNN + FFT prediction on raw image bytes."""
    tensor, img_array, _ = preprocess_image_bytes(img_bytes)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        cnn_score = float(cnn_model(tensor).squeeze())

    fft_score, _ = extract_fft_features(img_array)
    fft_norm     = min(fft_score * 2.5, 1.0)
    final_score  = CNN_WEIGHT * cnn_score + FFT_WEIGHT * fft_norm
    prediction   = "AI-Generated" if final_score >= THRESHOLD else "Real"
    denom        = (1 - THRESHOLD) if final_score >= THRESHOLD else THRESHOLD
    confidence   = min(abs(final_score - THRESHOLD) / (denom + 1e-8), 1.0)

    return {
        "prediction":       prediction,
        "fake_probability": round(final_score, 4),
        "real_probability": round(1 - final_score, 4),
        "confidence":       round(confidence, 4),
        "cnn_score":        round(cnn_score, 4),
        "fft_score":        round(fft_norm, 4),
        "threshold":        THRESHOLD,
    }


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service":    "Synthetic Media & Deepfake Detector",
        "version":    config['version'],
        "hybrid_auc": config['hybrid_auc'],
        "endpoints":  ["/detect/image", "/detect/video", "/detect/url"]
    })


@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    try:
        img_bytes          = request.files['image'].read()
        result             = predict_image(img_bytes)
        result['filename'] = request.files['image'].filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    try:
        video_file = request.files['video']
        suffix     = os.path.splitext(video_file.filename)[1] or '.mp4'
        tmp        = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path   = tmp.name
        tmp.close()
        video_file.save(tmp_path)

        frames  = extract_video_frames(tmp_path, max_frames=10)
        results = []

        for i, frame in enumerate(frames):
            pil_img = Image.fromarray(frame)
            buf     = io.BytesIO()
            pil_img.save(buf, format='PNG')
            res = predict_image(buf.getvalue())
            results.append({
                "frame":            i + 1,
                "fake_probability": res['fake_probability'],
                "prediction":       res['prediction']
            })

        os.remove(tmp_path)

        scores    = [r['fake_probability'] for r in results]
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))

        return jsonify({
            "filename":          video_file.filename,
            "frames_analyzed":   len(results),
            "avg_fake_score":    round(avg_score, 4),
            "max_fake_score":    round(max_score, 4),
            "overall_prediction":"AI-Generated" if avg_score >= THRESHOLD else "Real",
            "confidence":        round(abs(avg_score - 0.5) / 0.5, 4),
            "frame_results":     results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/url', methods=['POST'])
def detect_url():
    """Detect deepfake from image URL — used by Chrome extension."""
    import urllib.request

    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    try:
        req       = urllib.request.Request(
            data['url'],
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        img_bytes = urllib.request.urlopen(req, timeout=8).read()
        result    = predict_image(img_bytes)
        result['url'] = data['url']
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)