import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    """
    XceptionNet deepfake detector.
    Uses 'classifier' attribute name — matches Colab saved weights.
    Compatible with timm==0.9.16
    """
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            'legacy_xception',   # ✅ non-deprecated name
            pretrained=False,
            num_classes=0
        )

        # Detect feature dim safely at init time
        with torch.no_grad():
            dummy    = torch.randn(1, 3, 299, 299)
            feat_dim = self.backbone(dummy).shape[1]

        # ✅ Must be named 'classifier' — matches keys saved in Colab
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
        return self.classifier(self.backbone(x))


def load_model(weights_path: str, device: str = 'cpu') -> XceptionDetector:
    """
    Version-safe model loading.
    Handles both PyTorch >= 2.0 and < 2.0 API.
    """
    model = XceptionDetector().to(device)

    try:
        # PyTorch >= 2.0
        state = torch.load(
            weights_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # PyTorch < 2.0 — no weights_only param
        state = torch.load(weights_path, map_location=device)

    model.load_state_dict(state)
    model.eval()
    print(f"XceptionDetector loaded from {weights_path}")
    return model