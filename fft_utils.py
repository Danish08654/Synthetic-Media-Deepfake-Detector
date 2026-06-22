import numpy as np
import cv2

def extract_fft_features(img_array: np.ndarray):
    """
    FFT frequency anomaly detection.
    Version-safe — pure numpy/opencv, no version issues.
    """
    if img_array.ndim == 3:
        gray = cv2.cvtColor(
            img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_mag   = np.log1p(magnitude)
    mn, mx    = log_mag.min(), log_mag.max()
    log_norm  = (log_mag - mn) / (mx - mn + 1e-8)

    h, w         = gray.shape
    cy, cx       = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist         = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    max_d        = np.sqrt(cx**2 + cy**2)

    low_e  = float(log_norm[dist < max_d * 0.1].mean())
    high_e = float(log_norm[dist >= max_d * 0.4].mean())
    hf_ratio  = high_e / (low_e + high_e + 1e-8)
    row_var   = float(np.var(log_norm, axis=1).mean())
    col_var   = float(np.var(log_norm, axis=0).mean())
    periodic  = (row_var + col_var) / 2
    fft_score = hf_ratio * 0.6 + min(periodic, 1.0) * 0.4

    return float(fft_score), log_norm