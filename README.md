# Synthetic Media & Deepfake Detector

AI-powered system for detecting deepfake images, videos, and synthetic media using deep learning and frequency-domain analysis.

##  Features

- Deepfake face detection
- FFT-based frequency analysis
- XceptionNet inference with PyTorch
- Real-time Flask REST API
- Chrome extension for live webpage scanning
- Confidence score prediction

---

##  Tech Stack

- PyTorch
- XceptionNet
- OpenCV
- FFT Analysis
- Flask
- JavaScript (Chrome Extension)

---

##  Workflow

```text
Website Image → Chrome Extension → Flask API →
FFT + XceptionNet → REAL / FAKE Prediction

---

# Load Chrome Extension

Open chrome://extensions
Enable Developer Mode
Click Load Unpacked
Select the extension/ folder

---


#  Testing

Start Flask API
Load extension
Open a webpage with images
Extension scans and predicts
🟢 REAL
🔴 FAKE

---

#  Datasets
FaceForensics++
Celeb-DF
DFDC

---

#  Applications
Trust & Safety
Fake Media Detection
Social Media Moderation
Digital Forensics

---



# Author

Danish Zulfiqar
