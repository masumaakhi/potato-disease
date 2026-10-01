from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class DiseaseClass(BaseModel):
    id: int
    name: str
    code: str
    description: Optional[str] = None


class ClassProbability(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    percentage: str


class AttentionWeights(BaseModel):
    cnn_branch_weight: float = Field(..., description="Normalized attention weight allocated to CNN spatial branch")
    vit_branch_weight: float = Field(..., description="Normalized attention weight allocated to ViT semantic branch")
    dominant_branch: str = Field(..., description="Dominant architecture branch for this input ('CNN' or 'ViT')")


class InferenceInfo(BaseModel):
    inference_time_ms: float = Field(..., description="Forward pass inference latency in milliseconds")
    device: str = Field(..., description="Hardware compute device utilized (e.g. cpu, cuda)")
    input_resolution: str = Field(default="224x224", description="Model input resolution after preprocessing")
    timestamp: str = Field(..., description="ISO 8601 UTC execution timestamp")


class GradCAMResult(BaseModel):
    diagnostic_label: str = Field(
        default="Grad-CAM — CNN Branch Diagnostic View",
        description="Formal research diagnostic classification label"
    )
    disclaimer: str = Field(
        default="Diagnostic visualization of the CNN branch only. "
                "This does not represent an explanation of the entire CNN–ViT hybrid model "
                "or the Vision Transformer self-attention branch."
    )
    model_id: str
    model_name: str
    target_layer: str
    predicted_class: str
    predicted_class_id: int
    confidence: float
    target_class: str
    target_class_id: int
    target_class_probability: float
    gradcam_image_base64: str = Field(..., description="Alpha-blended heatmap overlay on original image in base64 JPEG format")
    heatmap_image_base64: Optional[str] = Field(None, description="Standalone colorized Jet heatmap in base64 JPEG format")
    heatmap_grid_size: Optional[List[int]] = Field(default=[7, 7], description="Spatial dimensions of target convolutional feature map")
    resolution: str = Field(default="224x224")
    alpha_blend: float = Field(default=0.45)


class PredictionResult(BaseModel):
    model_name: str
    model_id: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: List[ClassProbability]
    inference_time_ms: float
    inference_info: Optional[InferenceInfo] = None
    attention_weights: Optional[AttentionWeights] = None
    attention_data: Optional[Dict[str, Any]] = None
    gradcam_available: bool = False
    attention_weights_available: bool = False
    gradcam_image_base64: Optional[str] = None
    gradcam_view: Optional[GradCAMResult] = None
    specimen_warning: Optional[str] = None
    is_likely_potato_leaf: bool = True
    specimen_status: str = "optimal"


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    description: str
    parameters: Optional[str] = None
    checkpoint_available: bool = False
