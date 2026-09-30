import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import timm
from pathlib import Path
from typing import Optional, Union, Tuple


class AttentionGuidedCNNViTFusion(nn.Module):
    """
    Proposed Architecture: Attention-Guided Lightweight CNN-ViT Fusion Network
    Learns dynamic, sample-wise softmax-normalized branch attention weights [alpha_cnn, alpha_vit]
    to balance local CNN spatial features and global ViT self-attention.
    Total parameters: 10,109,249 (~10.11M)
    """
    def __init__(
        self,
        num_classes: int = 3,
        cnn_feature_dim: int = 1280,
        vit_feature_dim: int = 192,
        projection_dim: int = 256,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.35
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Branch
        base_cnn = efficientnet_b0(weights=None)
        self.cnn_features = base_cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # ViT Branch
        self.vit_backbone = timm.create_model(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=False,
            num_classes=0,
            fc_norm=True
        )
        self.vit_projection = nn.Sequential(
            nn.Linear(vit_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # Attention Gating Branch
        self.attention_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 2)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Feature extraction
        cnn_feat = self.cnn_pool(self.cnn_features(x)).flatten(1)
        vit_feat = self.vit_backbone(x)

        # 2. Linear projection & LayerNorm
        proj_cnn = self.cnn_projection(cnn_feat)
        proj_vit = self.vit_projection(vit_feat)

        # 3. Dynamic Attention Weights
        gate_in = torch.cat([proj_cnn, proj_vit], dim=1)
        gate_logits = self.attention_gate(gate_in)
        attn_weights = torch.softmax(gate_logits, dim=-1)

        # 4. Weighted Attention Fusion
        alpha_cnn = attn_weights[:, 0:1]
        alpha_vit = attn_weights[:, 1:2]
        fused = alpha_cnn * proj_cnn + alpha_vit * proj_vit

        # 5. Classification
        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights
        return logits


def load_attention_fusion(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    num_classes: int = 3,
    projection_dim: int = 256,
    classifier_hidden_dim: int = 256,
    dropout: float = 0.35
) -> AttentionGuidedCNNViTFusion:
    """
    Loads trained Attention-Guided CNN-ViT Fusion checkpoint strictly.
    Sets model to eval() mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = AttentionGuidedCNNViTFusion(
        num_classes=num_classes,
        projection_dim=projection_dim,
        classifier_hidden_dim=classifier_hidden_dim,
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
