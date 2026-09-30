import io
import base64
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.gradcam import gradcam_service, GradCAMCNNBranchOnlyError, GradCAMError
from app.services.inference import inference_service

client = TestClient(app)


def create_sample_leaf_bytes(color=(40, 160, 40), size=(256, 256)) -> bytes:
    """Creates an in-memory sample RGB leaf image as JPEG bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_gradcam_service_attention_fusion():
    """Grad-CAM on Attention-Guided CNN-ViT targets cnn_features[-1] and returns diagnostic view."""
    img_bytes = create_sample_leaf_bytes()
    res = gradcam_service.generate_gradcam(
        model_id="attention_fusion",
        image=img_bytes
    )

    assert res["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"
    assert "does not represent an explanation of the entire CNN–ViT hybrid model" in res["disclaimer"]
    assert "cnn_features[-1]" in res["target_layer"]
    assert res["model_id"] == "attention_fusion"
    assert res["predicted_class"] in ["Early Blight", "Late Blight", "Healthy"]
    assert 0.0 <= res["confidence"] <= 1.0

    # Verify base64 image decoding
    b64_img = res["gradcam_image_base64"]
    raw_overlay = base64.b64decode(b64_img)
    decoded_img = Image.open(io.BytesIO(raw_overlay))
    assert decoded_img.size == (224, 224)
    assert decoded_img.mode == "RGB"


def test_gradcam_service_efficientnet_b0():
    """Grad-CAM on baseline EfficientNet-B0 targets features[-1]."""
    img_bytes = create_sample_leaf_bytes()
    res = gradcam_service.generate_gradcam(
        model_id="efficientnet_b0",
        image=img_bytes
    )

    assert res["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"
    assert "features[-1]" in res["target_layer"]
    assert res["gradcam_image_base64"] is not None


def test_gradcam_service_cnn_vit_concat():
    """Grad-CAM on CNN-ViT concatenation targets cnn_features[-1]."""
    img_bytes = create_sample_leaf_bytes()
    res = gradcam_service.generate_gradcam(
        model_id="cnn_vit_concat",
        image=img_bytes
    )

    assert res["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"
    assert "cnn_features[-1]" in res["target_layer"]
    assert res["gradcam_image_base64"] is not None


def test_gradcam_deit_tiny_raises_cnn_only_error():
    """DeiT-Tiny is a pure Vision Transformer and must reject Grad-CAM."""
    img_bytes = create_sample_leaf_bytes()
    with pytest.raises(GradCAMCNNBranchOnlyError) as exc_info:
        gradcam_service.generate_gradcam(
            model_id="deit_tiny",
            image=img_bytes
        )

    err_msg = str(exc_info.value)
    assert "pure Vision Transformer" in err_msg
    assert "convolutional" in err_msg


def test_gradcam_specific_target_class():
    """Grad-CAM can compute activations for a specified disease class index."""
    img_bytes = create_sample_leaf_bytes()
    res = gradcam_service.generate_gradcam(
        model_id="attention_fusion",
        image=img_bytes,
        target_class_id=1  # Late Blight
    )

    assert res["target_class_id"] == 1
    assert res["target_class"] == "Late Blight"
    assert "gradcam_image_base64" in res


def test_api_predict_gradcam_endpoint_attention_fusion():
    """Dedicated POST /api/predict/gradcam returns structured diagnostic view."""
    img_bytes = create_sample_leaf_bytes()
    response = client.post(
        "/api/predict/gradcam",
        files={"image": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"
    assert "does not represent an explanation of the entire CNN–ViT hybrid model" in data["disclaimer"]
    assert data["model_id"] == "attention_fusion"
    assert "cnn_features[-1]" in data["target_layer"]
    assert data["gradcam_image_base64"] is not None


def test_api_predict_gradcam_endpoint_deit_tiny_rejected():
    """POST /api/predict/gradcam on pure ViT returns 400 with explanation."""
    img_bytes = create_sample_leaf_bytes()
    response = client.post(
        "/api/predict/gradcam",
        files={"image": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "deit_tiny"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "pure Vision Transformer" in detail
    assert "convolutional" in detail


def test_api_predict_with_explain_true():
    """POST /api/predict with explain=true automatically populates Grad-CAM for CNN models."""
    img_bytes = create_sample_leaf_bytes()
    response = client.post(
        "/api/predict",
        files={"image": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion", "explain": "true"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["gradcam_available"] is True
    assert data["gradcam_image_base64"] is not None
    assert data["gradcam_view"]["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"


def test_api_predict_with_explain_true_for_deit_tiny():
    """POST /api/predict with explain=true for deit_tiny leaves gradcam_available=False."""
    img_bytes = create_sample_leaf_bytes()
    response = client.post(
        "/api/predict",
        files={"image": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "deit_tiny", "explain": "true"}
    )

    assert response.status_code == 200
    data = response.json()
    # DeiT-Tiny has no CNN branch, so gradcam is not available
    assert data["gradcam_available"] is False
    assert data["gradcam_image_base64"] is None
