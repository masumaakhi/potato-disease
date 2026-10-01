import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import torch

from ..core.config import settings
from ..schemas.prediction import (
    PredictionResult,
    ClassProbability,
    InferenceInfo,
    AttentionWeights,
    GradCAMResult,
)
from ..models.efficientnet import load_efficientnet_b0
from ..models.deit import load_deit_tiny
from ..models.concat_fusion import load_concat_fusion
from ..models.attention_fusion import load_attention_fusion
from .preprocessing import preprocess_image
from .specimen_validator import evaluate_specimen_suitability

# Exact class mapping from dataloader_preprocessing_config.json
RESEARCH_CLASSES = [
    {"id": 0, "name": "Early Blight", "code": "Early_Blight"},
    {"id": 1, "name": "Late Blight", "code": "Late_Blight"},
    {"id": 2, "name": "Healthy", "code": "Healthy"},
]

MODEL_CHECKPOINT_DIRS = {
    "efficientnet_b0": "efficientnet_b0",
    "deit_tiny": "deit_tiny",
    "cnn_vit_concat": "cnn_vit_concat",
    "attention_fusion": "attention_fusion",
}

LOADER_DISPATCH = {
    "efficientnet_b0": load_efficientnet_b0,
    "deit_tiny": load_deit_tiny,
    "cnn_vit_concat": load_concat_fusion,
    "attention_fusion": load_attention_fusion,
}

MODEL_DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
    "cnn_vit_concat": "CNN–ViT Concatenation",
    "attention_fusion": "Attention-Guided CNN–ViT",
}


class ModelInferenceService:
    """
    Research Model Loading and Inference Layer.
    Manages loading of all four comparative architectures from actual trained checkpoints.
    All models are kept in eval() mode with torch.no_grad() execution.
    """
    def __init__(self):
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.device = settings.DEVICE

    def get_checkpoint_path(self, model_id: str, seed: int = 42) -> Optional[Path]:
        """Resolves the trained checkpoint file path for a model."""
        sub_dir = MODEL_CHECKPOINT_DIRS.get(model_id)
        if not sub_dir:
            return None
        folder = settings.CHECKPOINTS_DIR / sub_dir
        if not folder.exists():
            return None

        # 1. Preferred seed (e.g. *best_seed42.pth)
        preferred = list(folder.glob(f"*best_seed{seed}.pth"))
        if preferred:
            return preferred[0]

        # 2. Any best checkpoint (*best*.pth)
        any_best = list(folder.glob("*best*.pth"))
        if any_best:
            return any_best[0]

        # 3. Any valid weight file (.pth / .pt)
        all_weights = list(folder.glob("*.pth")) + list(folder.glob("*.pt"))
        if all_weights:
            return all_weights[0]

        return None

    def is_checkpoint_available(self, model_id: str, seed: int = 42) -> bool:
        """Checks if the weight checkpoint for a given model is present."""
        return self.get_checkpoint_path(model_id, seed=seed) is not None

    def get_or_load_model(self, model_id: str, seed: int = 42) -> torch.nn.Module:
        """
        Loads and caches a model by ID using strict checkpoint loading.
        Raises descriptive error if checkpoint is missing or fails to load.
        """
        cache_key = f"{model_id}_seed{seed}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        ckpt_path = self.get_checkpoint_path(model_id, seed=seed)
        if not ckpt_path:
            raise FileNotFoundError(
                f"Trained checkpoint for '{model_id}' (seed {seed}) not found."
            )

        loader_fn = LOADER_DISPATCH.get(model_id)
        if not loader_fn:
            raise ValueError(f"No model loader defined for '{model_id}'.")

        try:
            model = loader_fn(
                checkpoint_path=ckpt_path,
                device=self.device,
                num_classes=len(RESEARCH_CLASSES)
            )
            model.eval()
            # Freeze parameters: saves RAM and speeds up execution by avoiding gradient buffers
            for param in model.parameters():
                param.requires_grad = False
            self.loaded_models[cache_key] = model
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint for model '{model_id}': {str(exc)}"
            ) from exc

    def predict(
        self,
        model_id: str,
        image: Image.Image,
        seed: int = 42,
        explain: bool = False
    ) -> PredictionResult:
        """
        Executes prediction on an input PIL image using the loaded model.
        Evaluation mode and torch.inference_mode() are strictly enforced for speed.
        """
        model = self.get_or_load_model(model_id, seed=seed)

        # Preprocessing aligned with research dataloader configuration
        input_tensor = preprocess_image(image).to(self.device)

        start_time = time.perf_counter()
        attention_weights_obj: Optional[AttentionWeights] = None
        attention_data: Optional[Dict[str, Any]] = None

        with torch.inference_mode():
            if model_id == "attention_fusion":
                logits, attn_weights = model(input_tensor, return_attention=True)
                attn_cnn = float(attn_weights[0, 0].item())
                attn_vit = float(attn_weights[0, 1].item())
                dominant = "ViT" if attn_vit > attn_cnn else "CNN"
                attention_weights_obj = AttentionWeights(
                    cnn_branch_weight=round(attn_cnn, 4),
                    vit_branch_weight=round(attn_vit, 4),
                    dominant_branch=dominant
                )
                attention_data = {
                    "cnn_branch_weight": round(attn_cnn, 4),
                    "vit_branch_weight": round(attn_vit, 4),
                    "dominant_branch": dominant
                }
            else:
                logits = model(input_tensor)

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            conf, pred_idx = torch.max(probs, dim=0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        probabilities: List[ClassProbability] = []
        for cls_info in RESEARCH_CLASSES:
            prob_val = float(probs[cls_info["id"]].item())
            probabilities.append(
                ClassProbability(
                    class_name=cls_info["name"],
                    probability=round(prob_val, 4),
                    percentage=f"{prob_val * 100:.2f}%"
                )
            )

        predicted_class_name = RESEARCH_CLASSES[pred_idx.item()]["name"]

        inference_info = InferenceInfo(
            inference_time_ms=round(elapsed_ms, 2),
            device=str(self.device),
            input_resolution="224x224",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Grad-CAM CNN Branch Diagnostic View generation when explain=True
        gradcam_res: Optional[GradCAMResult] = None
        gradcam_available = False
        gradcam_b64: Optional[str] = None

        if explain and model_id != "deit_tiny":
            try:
                from .gradcam import gradcam_service
                gcam_dict = gradcam_service.generate_gradcam(
                    model_id=model_id,
                    image=image,
                    target_class_id=pred_idx.item(),
                    seed=seed
                )
                gradcam_res = GradCAMResult(**gcam_dict)
                gradcam_available = True
                gradcam_b64 = gcam_dict["gradcam_image_base64"]
            except Exception:
                gradcam_available = False
                gradcam_res = None
                gradcam_b64 = None

        # Specimen Suitability & Out-of-Distribution Quality Check
        specimen_check = evaluate_specimen_suitability(image, top_confidence=float(conf.item()))

        return PredictionResult(
            model_name=MODEL_DISPLAY_NAMES.get(model_id, model_id),
            model_id=model_id,
            predicted_class=predicted_class_name,
            confidence=round(float(conf.item()), 4),
            probabilities=probabilities,
            inference_time_ms=round(elapsed_ms, 2),
            inference_info=inference_info,
            attention_weights=attention_weights_obj,
            attention_data=attention_data,
            gradcam_available=gradcam_available,
            attention_weights_available=(attention_weights_obj is not None),
            gradcam_image_base64=gradcam_b64,
            gradcam_view=gradcam_res,
            specimen_warning=specimen_check.get("specimen_warning"),
            is_likely_potato_leaf=specimen_check.get("is_likely_potato_leaf", True),
            specimen_status=specimen_check.get("specimen_status", "optimal")
        )


inference_service = ModelInferenceService()
