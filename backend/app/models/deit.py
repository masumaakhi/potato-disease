import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional, Union


class DeiTTinyClassifier(nn.Module):
    """
    Compact Vision Transformer Baseline: DeiT-Tiny
    Utilized for global contextual representation and long-range semantic dependencies.
    Total parameters: 5,524,995 (~5.52M)
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_deit_tiny(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    num_classes: int = 3
) -> nn.Module:
    """
    Loads trained DeiT-Tiny checkpoint strictly.
    Sets model to eval() mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=False,
        num_classes=num_classes
    )
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
