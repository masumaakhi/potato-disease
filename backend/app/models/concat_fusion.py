import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import timm
from pathlib import Path
from typing import Optional, Union


class CNNViTConcatFusion(nn.Module):
    """
    Baseline Fusion Architecture: Direct Feature Concatenation
    Combines pooled local CNN features (1280) and global ViT tokens (192) -> (1472).
    Total parameters: 10,290,623 (~10.29M)
    """
    def __init__(
        self,
        num_classes: int = 3,
        cnn_feature_dim: int = 1280,
        vit_feature_dim: int = 192,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.35
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Branch
        base_cnn = efficientnet_b0(weights=None)
        self.cnn_features = base_cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # ViT Branch
        self.vit_backbone = timm.create_model(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=False,
            num_classes=0,
            fc_norm=True
        )

        concatenated_dim = cnn_feature_dim + vit_feature_dim  # 1472

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(concatenated_dim),
            nn.Dropout(dropout),
            nn.Linear(concatenated_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_pool(self.cnn_features(x)).flatten(1)
        vit_feat = self.vit_backbone(x)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(fused)


def load_concat_fusion(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    num_classes: int = 3,
    dropout: float = 0.35,
    fusion_hidden_dim: int = 512
) -> CNNViTConcatFusion:
    """
    Loads trained CNN-ViT Concatenation checkpoint strictly.
    Sets model to eval() mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = CNNViTConcatFusion(
        num_classes=num_classes,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout
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
