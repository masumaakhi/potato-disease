import io
import base64
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib

from ..core.config import settings
from .preprocessing import preprocess_image, validate_and_load_image
from .inference import inference_service, RESEARCH_CLASSES, MODEL_DISPLAY_NAMES


class GradCAMError(Exception):
    """Base exception for Grad-CAM errors."""
    pass


class GradCAMCNNBranchOnlyError(GradCAMError):
    """Raised when Grad-CAM is requested on an architecture without a CNN branch (e.g. pure ViT)."""
    pass


class GradCAMService:
    """
    Research Grad-CAM Explainability Service.
    
    IMPORTANT RESEARCH METHODOLOGY NOTE:
    In the research paper, Grad-CAM is utilized strictly as a diagnostic visualization
    of the CNN branch (EfficientNet-B0 backbone) to inspect local convolutional feature
    activations and lesion localization.
    
    It is NOT an explanation of the complete CNN–ViT hybrid model, nor does it capture
    the multi-head self-attention mechanisms of the DeiT-Tiny Vision Transformer branch.
    """

    DIAGNOSTIC_LABEL = "Grad-CAM — CNN Branch Diagnostic View"
    DISCLAIMER = (
        "Diagnostic visualization of the CNN branch only. "
        "This does not represent an explanation of the entire CNN–ViT hybrid model "
        "or the Vision Transformer self-attention branch."
    )

    def __init__(self):
        self.device = settings.DEVICE
        # Cache colormap reference
        self.colormap = matplotlib.colormaps["jet"]

    def get_cnn_target_layer(self, model: nn.Module, model_id: str) -> Tuple[nn.Module, str]:
        """
        Inspects the actual model architecture and retrieves the final convolutional layer
        of the CNN branch.
        
        - Proposed Attention-Guided Fusion: model.cnn_features[-1]
        - CNN-ViT Concatenation Fusion: model.cnn_features[-1]
        - Standalone EfficientNet-B0: model.features[-1]
        - DeiT-Tiny: No CNN branch -> Raises GradCAMCNNBranchOnlyError
        """
        if model_id == "deit_tiny":
            raise GradCAMCNNBranchOnlyError(
                "Grad-CAM is only applicable to models with a convolutional branch. "
                "DeiT-Tiny is a pure Vision Transformer (ViT) architecture without convolutional feature maps. "
                "For transformer attention analysis, use patch self-attention inspection."
            )

        # 1. Proposed and Concatenation fusion models: self.cnn_features = base_cnn.features
        if hasattr(model, "cnn_features") and isinstance(model.cnn_features, nn.Sequential):
            target = model.cnn_features[-1]
            layer_name = "cnn_features[-1] (Conv2dNormActivation: 1280 channels)"
            return target, layer_name

        # 2. Standalone EfficientNet-B0: self.features
        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            target = model.features[-1]
            layer_name = "features[-1] (Conv2dNormActivation: 1280 channels)"
            return target, layer_name

        raise GradCAMError(
            f"Unable to locate CNN feature extraction layer on model architecture for '{model_id}'."
        )

    def generate_gradcam(
        self,
        model_id: str,
        image: Union[Image.Image, bytes, str],
        target_class_id: Optional[int] = None,
        seed: int = 42,
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Executes Grad-CAM for the CNN branch on the specified model and input image.
        
        Args:
            model_id: One of 'attention_fusion', 'cnn_vit_concat', 'efficientnet_b0'
            image: PIL Image or raw bytes
            target_class_id: Optional target class (if None, uses predicted class argmax)
            seed: Checkpoint random seed (default 42)
            alpha: Heatmap blend weight (0.45 matching research notebook)
            
        Returns:
            Dict containing diagnostic label, disclaimer, target layer, class prediction,
            and base64 encoded overlay and heatmap images.
        """
        # 1. Validate image and prepare preprocessed input tensor
        pil_image = validate_and_load_image(image)
        input_tensor = preprocess_image(pil_image, device=self.device)

        # 2. Retrieve model checkpoint and ensure eval mode
        model = inference_service.get_or_load_model(model_id, seed=seed)
        model.eval()

        # 3. Resolve target convolutional layer
        target_layer, layer_name = self.get_cnn_target_layer(model, model_id)

        # 4. Register forward and backward hooks for activation and gradient extraction
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Enable gradients specifically for the backward pass
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                
                # Forward pass
                if model_id == "attention_fusion":
                    logits, _ = model(input_tensor, return_attention=True)
                else:
                    out = model(input_tensor)
                    logits = out[0] if isinstance(out, tuple) else out

                # Probabilities & predicted class
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                predicted_class_id = int(logits.argmax(dim=1)[0])

                if target_class_id is None:
                    target_class_id = predicted_class_id

                if not (0 <= target_class_id < len(RESEARCH_CLASSES)):
                    raise ValueError(f"Target class ID {target_class_id} is out of bounds (0-2).")

                # Backward pass for the target class score
                score = logits[0, target_class_id]
                score.backward()

            # Ensure hooks captured data
            if not activations or not gradients:
                raise RuntimeError("Failed to capture feature activations or gradients from target layer.")

            act = activations[0]  # Shape: (1, 1280, 7, 7)
            grad = gradients[0]   # Shape: (1, 1280, 7, 7)

            # Global average pooling of gradients over spatial dimensions (H, W) -> weights (1, C, 1, 1)
            weights = grad.mean(dim=(2, 3), keepdim=True)

            # Linear combination of forward feature maps weighted by gradient importance
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True)).squeeze()  # (7, 7)

            # Min-max normalization [0.0, 1.0]
            cam_min = cam.min()
            cam_max = cam.max()
            cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            cam_np = cam_norm.detach().cpu().numpy().astype(np.float32)

        finally:
            # Safely unregister hooks and clean up gradient buffers
            fh.remove()
            bh.remove()
            model.zero_grad(set_to_none=True)

        # 5. Render heatmap and blended overlay image
        cam_pil = Image.fromarray(np.uint8(np.clip(cam_np, 0.0, 1.0) * 255.0)).resize(
            (224, 224), resample=Image.Resampling.BILINEAR
        )
        cam_resized_norm = np.asarray(cam_pil, dtype=np.float32) / 255.0

        # Apply Jet colormap (matching research publication figures)
        heatmap_rgba = self.colormap(cam_resized_norm)  # (224, 224, 4)
        heatmap_rgb = heatmap_rgba[:, :, :3].astype(np.float32)  # (224, 224, 3)

        # Prepare normalized display image resized to 224x224
        disp_image = np.asarray(pil_image.resize((224, 224)), dtype=np.float32) / 255.0
        if disp_image.ndim == 2:
            disp_image = np.stack([disp_image] * 3, axis=-1)
        elif disp_image.shape[-1] == 4:
            disp_image = disp_image[:, :, :3]

        # Alpha-blend overlay: (1 - alpha) * original + alpha * heatmap
        blended = (1.0 - alpha) * disp_image + alpha * heatmap_rgb
        overlay_uint8 = np.uint8(np.clip(blended * 255.0, 0.0, 255.0))
        heatmap_uint8 = np.uint8(np.clip(heatmap_rgb * 255.0, 0.0, 255.0))

        # 6. Encode images to base64 JPEG format
        overlay_b64 = self._pil_to_base64_jpeg(Image.fromarray(overlay_uint8))
        heatmap_b64 = self._pil_to_base64_jpeg(Image.fromarray(heatmap_uint8))

        target_class_meta = RESEARCH_CLASSES[target_class_id]

        return {
            "diagnostic_label": self.DIAGNOSTIC_LABEL,
            "disclaimer": self.DISCLAIMER,
            "model_id": model_id,
            "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
            "target_layer": layer_name,
            "predicted_class": RESEARCH_CLASSES[predicted_class_id]["name"],
            "predicted_class_id": predicted_class_id,
            "confidence": float(probs[predicted_class_id]),
            "target_class": target_class_meta["name"],
            "target_class_id": target_class_id,
            "target_class_probability": float(probs[target_class_id]),
            "gradcam_image_base64": overlay_b64,
            "heatmap_image_base64": heatmap_b64,
            "heatmap_grid_size": list(cam_np.shape),
            "resolution": "224x224",
            "alpha_blend": alpha
        }

    @staticmethod
    def _pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
        """Encodes PIL Image into Base64 JPEG string."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Service instance singleton
gradcam_service = GradCAMService()
