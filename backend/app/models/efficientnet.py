import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from pathlib import Path
from typing import Optional, Union


class EfficientNetB0Classifier(nn.Module):
    """
    Lightweight CNN Baseline: EfficientNet-B0
    Utilized for local spatial texture and disease lesion pattern extraction.
    Total parameters: 4,011,391 (~4.01M)
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_efficientnet_b0(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    num_classes: int = 3
) -> EfficientNetB0Classifier:
    """
    Loads trained EfficientNet-B0 checkpoint strictly.
    Sets model to eval() mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = EfficientNetB0Classifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
