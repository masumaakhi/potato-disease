from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any

from ..schemas.prediction import PredictionResult, GradCAMResult
from ..services.inference import inference_service, MODEL_DISPLAY_NAMES
from ..services.preprocessing import (
    validate_and_load_image,
    ImagePreprocessingError,
    CorruptImageError,
    EmptyImageError,
    UnsupportedImageFormatError,
    get_preprocessing_pipeline_info
)
from ..services.gradcam import (
    gradcam_service,
    GradCAMCNNBranchOnlyError,
    GradCAMError
)

router = APIRouter(prefix="/predict", tags=["Prediction"])

SUPPORTED_MODELS = list(MODEL_DISPLAY_NAMES.keys())


@router.get("/status")
def prediction_pipeline_status() -> Dict[str, Any]:
    """
    Returns current prediction pipeline status, supported model architectures,
    and active research preprocessing configuration.
    """
    prep_info = get_preprocessing_pipeline_info()
    return {
        "status": "ready",
        "supported_models": SUPPORTED_MODELS,
        "model_names": MODEL_DISPLAY_NAMES,
        "preprocessing": prep_info
    }


@router.post("", response_model=PredictionResult)
async def predict_leaf_disease(
    image: Optional[UploadFile] = File(None, description="Uploaded potato leaf image (JPEG, PNG, WEBP, BMP, TIFF)"),
    file: Optional[UploadFile] = File(None, description="Alternative field for uploaded image"),
    model: Optional[str] = Form(None, description="Model identifier: efficientnet_b0, deit_tiny, cnn_vit_concat, attention_fusion"),
    model_id: Optional[str] = Form(None, description="Alternative field for model identifier"),
    explain: Optional[bool] = Form(True, description="Return model interpretability features (e.g. Grad-CAM and attention weights)")
):
    """
    Execute real potato leaf disease classification using the requested trained model.

    Supported models:
    - **efficientnet_b0**: Lightweight CNN Baseline
    - **deit_tiny**: Compact Vision Transformer Baseline
    - **cnn_vit_concat**: CNN–ViT Concatenation Fusion
    - **attention_fusion**: Proposed Attention-Guided CNN–ViT Network

    Returns:
    - **predicted_class**: Classification label (Early Blight, Late Blight, Healthy)
    - **confidence**: Top prediction confidence score [0.0 - 1.0]
    - **probabilities**: Per-class probability distribution
    - **model_name**: Formal research architecture title
    - **inference_info**: Latency in ms, compute device, resolution, timestamp
    - **attention_weights**: Dynamic CNN and ViT branch attention weights (for attention_fusion)
    - **gradcam_image_base64**: Diagnostic Grad-CAM overlay for models with a CNN branch (when explain=True)
    """
    # 1. Resolve uploaded image file (supports 'image' or 'file')
    uploaded_file = image or file
    if uploaded_file is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required image file. Please provide an image using the 'image' or 'file' field."
        )

    # 2. Resolve selected model architecture (supports 'model' or 'model_id')
    selected_model = (model or model_id or "").strip()
    if not selected_model:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required 'model' parameter. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    if selected_model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selected_model}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # 3. Validate and load image
    try:
        raw_bytes = await uploaded_file.read()
        pil_image = validate_and_load_image(raw_bytes)
    except EmptyImageError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except UnsupportedImageFormatError as err:
        raise HTTPException(status_code=415, detail=str(err))
    except CorruptImageError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except ImagePreprocessingError as err:
        raise HTTPException(status_code=400, detail=f"Image validation error: {str(err)}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image input data.")

    # 4. Execute inference with the actual trained model checkpoint
    try:
        result = inference_service.predict(
            model_id=selected_model,
            image=pil_image,
            explain=bool(explain)
        )
        return result
    except FileNotFoundError:
        # Do not expose internal filesystem paths
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint for '{selected_model}' is not available."
        )
    except Exception:
        # Generic sanitized error response without leaking stack traces or local paths
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute inference for model '{selected_model}'."
        )


@router.post("/gradcam", response_model=GradCAMResult)
async def generate_cnn_gradcam_view(
    image: Optional[UploadFile] = File(None, description="Uploaded potato leaf image (JPEG, PNG, WEBP, BMP, TIFF)"),
    file: Optional[UploadFile] = File(None, description="Alternative field for uploaded image"),
    model: Optional[str] = Form("attention_fusion", description="Model architecture with CNN branch"),
    model_id: Optional[str] = Form(None, description="Alternative field for model identifier"),
    target_class_id: Optional[int] = Form(None, description="Optional target class ID (0: Early Blight, 1: Late Blight, 2: Healthy). If omitted, uses predicted class.")
):
    """
    Generate a Grad-CAM visualization for the CNN branch of the requested model.

    IMPORTANT:
    This endpoint generates a **Grad-CAM — CNN Branch Diagnostic View**.
    It is used strictly as a diagnostic tool for convolutional feature maps and lesion
    localization. It does NOT represent an explanation of the complete hybrid network
    or the Vision Transformer self-attention branch.
    """
    # 1. Resolve uploaded image file
    uploaded_file = image or file
    if uploaded_file is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required image file. Please provide an image using the 'image' or 'file' field."
        )

    # 2. Resolve selected model architecture
    selected_model = (model_id or model or "attention_fusion").strip()
    if selected_model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selected_model}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # 3. Validate image
    try:
        raw_bytes = await uploaded_file.read()
        pil_image = validate_and_load_image(raw_bytes)
    except EmptyImageError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except UnsupportedImageFormatError as err:
        raise HTTPException(status_code=415, detail=str(err))
    except CorruptImageError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except ImagePreprocessingError as err:
        raise HTTPException(status_code=400, detail=f"Image validation error: {str(err)}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image input data.")

    # 4. Generate Grad-CAM for CNN branch
    try:
        gcam_dict = gradcam_service.generate_gradcam(
            model_id=selected_model,
            image=pil_image,
            target_class_id=target_class_id
        )
        return GradCAMResult(**gcam_dict)
    except GradCAMCNNBranchOnlyError as err:
        raise HTTPException(
            status_code=400,
            detail=str(err)
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint for '{selected_model}' is not available."
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Grad-CAM visualization for '{selected_model}'."
        )
