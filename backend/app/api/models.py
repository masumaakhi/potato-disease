from fastapi import APIRouter
from typing import List
from ..schemas.prediction import ModelInfo
from ..services.inference import inference_service

router = APIRouter(prefix="/models", tags=["Models"])

SUPPORTED_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="efficientnet_b0",
        name="EfficientNet-B0",
        architecture="Lightweight CNN Baseline",
        description="Convolutional architecture focused on local spatial textures and fine disease lesions.",
        parameters="4.01M (4,011,391)",
        checkpoint_available=False,
    ),
    ModelInfo(
        id="deit_tiny",
        name="DeiT-Tiny",
        architecture="Compact Vision Transformer Baseline",
        description="Self-attention transformer architecture capturing global contextual relationships.",
        parameters="5.52M (5,524,995)",
        checkpoint_available=False,
    ),
    ModelInfo(
        id="cnn_vit_concat",
        name="CNN–ViT Concatenation",
        architecture="Feature Concatenation Fusion",
        description="Direct concatenation of pooled CNN feature vectors and ViT class tokens.",
        parameters="10.29M (10,290,623)",
        checkpoint_available=False,
    ),
    ModelInfo(
        id="attention_fusion",
        name="Attention-Guided CNN–ViT",
        architecture="Proposed Attention-Guided Network",
        description="Adaptive attention-weighted fusion aligning local CNN features with global ViT representations.",
        parameters="10.11M (10,109,249)",
        checkpoint_available=False,
    ),
]


@router.get("", response_model=List[ModelInfo])
def get_models() -> List[ModelInfo]:
    """
    Retrieve metadata and checkpoint availability for all 4 research models:
    - EfficientNet-B0
    - DeiT-Tiny
    - CNN–ViT Concatenation
    - Attention-Guided CNN–ViT
    """
    results: List[ModelInfo] = []
    for model in SUPPORTED_MODELS:
        model_copy = model.model_copy()
        model_copy.checkpoint_available = inference_service.is_checkpoint_available(model.id)
        results.append(model_copy)
    return results


@router.get("/{model_id}", response_model=ModelInfo)
def get_model_by_id(model_id: str) -> ModelInfo:
    """Retrieve details for a specific model architecture by its ID."""
    for model in SUPPORTED_MODELS:
        if model.id == model_id:
            model_copy = model.model_copy()
            model_copy.checkpoint_available = inference_service.is_checkpoint_available(model.id)
            return model_copy
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
