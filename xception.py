import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    """
    XceptionNet deepfake detector.
    Compatible with timm and deployment environments.
    """

    def __init__(self):
        super().__init__()

        # Backbone model
        self.backbone = timm.create_model(
            'legacy_xception',
            pretrained=False,
            num_classes=0
        )

        # Infer feature dimension safely
        with torch.no_grad():
            dummy = torch.randn(1, 3, 299, 299)
            feat_dim = self.backbone(dummy).shape[1]

        # Classifier head (must match training)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# =========================================================
# 🚀 SAFE MODEL LOADING (FIXES STREAMLIT + PYTORCH ISSUES)
# =========================================================

def load_model(weights_path: str, device: str = 'cpu') -> XceptionDetector:
    """
    Robust model loader:
    - Handles PyTorch 2.x strict loading
    - Handles older checkpoints
    - Prevents UnpicklingError in Streamlit Cloud
    """

    model = XceptionDetector().to(device)

    try:
        # Try modern safe loading first
        state = torch.load(
            weights_path,
            map_location=device,
            weights_only=True
        )

    except Exception as e:
        print(f"[WARN] Safe load failed: {e}")
        print("[INFO] Trying legacy torch.load...")

        # Fallback for older / full object checkpoints
        state = torch.load(
            weights_path,
            map_location=device,
            weights_only=False
        )

    # If checkpoint contains extra metadata, extract weights safely
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"XceptionDetector loaded successfully from: {weights_path}")

    return model
