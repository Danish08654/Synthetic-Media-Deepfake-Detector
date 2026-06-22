import numpy as np
import cv2
import io
from PIL import Image
import torchvision.transforms as T
import torch

TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def preprocess_image_bytes(image_bytes: bytes):
    """Convert raw bytes → PIL image, numpy array, and tensor."""
    pil_img   = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(pil_img.resize((299, 299)))
    tensor    = TRANSFORM(pil_img).unsqueeze(0)
    return tensor, img_array, pil_img


def extract_video_frames(video_path: str, max_frames: int = 10):
    """Extract evenly spaced frames from video file."""
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, total // max_frames)
    frames = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames