import io
import base64
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.inference import inference_service, RESEARCH_CLASSES
from app.services.preprocessing import load_preprocessing_config

client = TestClient(app)


def make_test_leaf_image(color=(50, 150, 50), size=(256, 256)) -> bytes:
    """Creates a deterministic synthetic leaf image as JPEG bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ==============================================================================
# 1. GET /health
# ==============================================================================
def test_01_get_health():
    """Verify GET /health and GET /api/health return healthy status and checkpoints."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == 200, f"Health check failed on {path}"
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data
        assert "checkpoints_available" in data
        assert data["checkpoints_available"] is True


# ==============================================================================
# 2. GET /models
# ==============================================================================
def test_02_get_models():
    """Verify GET /api/models returns all 4 research models with checkpoints available."""
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 4

    model_ids = {m["id"] for m in models}
    expected_ids = {"efficientnet_b0", "deit_tiny", "cnn_vit_concat", "attention_fusion"}
    assert model_ids == expected_ids

    for m in models:
        assert m["checkpoint_available"] is True, f"Checkpoint not found for {m['id']}"
        assert m["name"]
        assert m["architecture"]


# ==============================================================================
# 3. POST /predict with EfficientNet-B0
# ==============================================================================
def test_03_predict_efficientnet_b0():
    """Verify real inference using EfficientNet-B0 checkpoint."""
    img_bytes = make_test_leaf_image(color=(40, 140, 40))
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "efficientnet_b0"}
    )
    assert response.status_code == 200
    res = response.json()

    assert res["model_id"] == "efficientnet_b0"
    assert res["model_name"] == "EfficientNet-B0"
    assert res["predicted_class"] in [c["name"] for c in RESEARCH_CLASSES]
    assert 0.0 <= res["confidence"] <= 1.0

    # Probabilities sum to ~1.0
    probs_sum = sum(p["probability"] for p in res["probabilities"])
    assert pytest.approx(probs_sum, abs=0.01) == 1.0

    # Class ordering matches research configuration exactly
    for idx, expected in enumerate(RESEARCH_CLASSES):
        assert res["probabilities"][idx]["class_name"] == expected["name"]

    # Standalone CNN has no fusion attention weights
    assert res["attention_weights"] is None
    assert res["attention_weights_available"] is False


# ==============================================================================
# 4. POST /predict with DeiT-Tiny
# ==============================================================================
def test_04_predict_deit_tiny():
    """Verify real inference using DeiT-Tiny pure Vision Transformer checkpoint."""
    img_bytes = make_test_leaf_image(color=(60, 160, 60))
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "deit_tiny"}
    )
    assert response.status_code == 200
    res = response.json()

    assert res["model_id"] == "deit_tiny"
    assert res["model_name"] == "DeiT-Tiny"
    assert res["predicted_class"] in [c["name"] for c in RESEARCH_CLASSES]
    assert 0.0 <= res["confidence"] <= 1.0

    probs_sum = sum(p["probability"] for p in res["probabilities"])
    assert pytest.approx(probs_sum, abs=0.01) == 1.0

    # DeiT-Tiny is pure ViT -> no attention fusion weights and no CNN Grad-CAM
    assert res["attention_weights"] is None
    assert res["gradcam_available"] is False
    assert res["gradcam_image_base64"] is None


# ==============================================================================
# 5. POST /predict with CNN–ViT Concatenation
# ==============================================================================
def test_05_predict_cnn_vit_concat():
    """Verify real inference using CNN-ViT Concatenation checkpoint."""
    img_bytes = make_test_leaf_image(color=(70, 150, 50))
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "cnn_vit_concat"}
    )
    assert response.status_code == 200
    res = response.json()

    assert res["model_id"] == "cnn_vit_concat"
    assert res["model_name"] == "CNN–ViT Concatenation"
    assert res["predicted_class"] in [c["name"] for c in RESEARCH_CLASSES]
    assert 0.0 <= res["confidence"] <= 1.0

    probs_sum = sum(p["probability"] for p in res["probabilities"])
    assert pytest.approx(probs_sum, abs=0.01) == 1.0

    # Concatenation has no dynamic attention weights
    assert res["attention_weights"] is None


# ==============================================================================
# 6. POST /predict with Attention-Guided CNN–ViT
# ==============================================================================
def test_06_predict_attention_fusion():
    """Verify real inference using proposed Attention-Guided CNN-ViT checkpoint."""
    img_bytes = make_test_leaf_image(color=(55, 145, 55))
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion"}
    )
    assert response.status_code == 200
    res = response.json()

    assert res["model_id"] == "attention_fusion"
    assert res["model_name"] == "Attention-Guided CNN–ViT"
    assert res["predicted_class"] in [c["name"] for c in RESEARCH_CLASSES]
    assert 0.0 <= res["confidence"] <= 1.0

    probs_sum = sum(p["probability"] for p in res["probabilities"])
    assert pytest.approx(probs_sum, abs=0.01) == 1.0


# ==============================================================================
# 7. Attention weights for the attention model
# ==============================================================================
def test_07_attention_weights_properties():
    """Verify attention weights are normalized, dynamic, and sum to ~1.0."""
    img_bytes = make_test_leaf_image(color=(50, 130, 70))
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion"}
    )
    assert response.status_code == 200
    res = response.json()

    attn = res["attention_weights"]
    assert attn is not None, "Attention weights must be present for attention_fusion"
    assert res["attention_weights_available"] is True

    cw = attn["cnn_branch_weight"]
    vw = attn["vit_branch_weight"]
    assert 0.0 <= cw <= 1.0
    assert 0.0 <= vw <= 1.0
    assert pytest.approx(cw + vw, abs=0.01) == 1.0
    assert attn["dominant_branch"] in ["CNN", "ViT"]


# ==============================================================================
# 8. Grad-CAM for the CNN branch
# ==============================================================================
def test_08_gradcam_cnn_branch():
    """Verify Grad-CAM is strictly labeled as CNN Branch Diagnostic View and targets conv layer."""
    img_bytes = make_test_leaf_image(color=(45, 135, 45))

    # Dedicated endpoint test
    response = client.post(
        "/api/predict/gradcam",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion"}
    )
    assert response.status_code == 200
    gcam = response.json()

    # Exact diagnostic label
    assert gcam["diagnostic_label"] == "Grad-CAM — CNN Branch Diagnostic View"
    # Strict disclaimer
    assert "does not represent an explanation of the entire CNN–ViT hybrid model" in gcam["disclaimer"]
    # Actual layer from EfficientNet-B0 backbone
    assert "cnn_features[-1]" in gcam["target_layer"]

    # Valid base64 overlay image decoding
    overlay_bytes = base64.b64decode(gcam["gradcam_image_base64"])
    overlay_img = Image.open(io.BytesIO(overlay_bytes))
    assert overlay_img.size == (224, 224)

    # Valid base64 heatmap decoding
    heatmap_bytes = base64.b64decode(gcam["heatmap_image_base64"])
    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
    assert heatmap_img.size == (224, 224)


# ==============================================================================
# 9. No Fake/Random Predictions (Determinism Verification)
# ==============================================================================
def test_09_inference_determinism():
    """Verify real checkpoints produce identical predictions across repeat inferences."""
    img_bytes = make_test_leaf_image(color=(48, 142, 52))

    resp1 = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion", "explain": "false"}
    )
    resp2 = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "attention_fusion", "explain": "false"}
    )

    data1 = resp1.json()
    data2 = resp2.json()

    assert data1["predicted_class"] == data2["predicted_class"]
    assert data1["confidence"] == data2["confidence"]
    assert data1["probabilities"] == data2["probabilities"]
    assert data1["attention_weights"] == data2["attention_weights"]


# ==============================================================================
# 10. Preprocessing Alignment with Research Configuration
# ==============================================================================
def test_10_preprocessing_configuration_match():
    """Verify preprocessing matches dataloader_preprocessing_config.json exactly."""
    config = load_preprocessing_config()

    assert config["image_size"] == 224
    assert config["resize_size"] == 256
    assert config["normalization_mean"] == [0.485, 0.456, 0.406]
    assert config["normalization_std"] == [0.229, 0.224, 0.225]
    assert config["class_names"] == ["Early_Blight", "Late_Blight", "Healthy"]


# ==============================================================================
# 11. Error Handling & Path Privacy
# ==============================================================================
def test_11_error_handling_and_no_path_leak():
    """Verify input validation, rejection of DeiT for Grad-CAM, and no path leaks."""
    # 1. Missing image
    r = client.post("/api/predict", data={"model": "attention_fusion"})
    assert r.status_code == 400

    # 2. Unsupported model
    img_bytes = make_test_leaf_image()
    r = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "non_existent_model"}
    )
    assert r.status_code == 400

    # 3. Corrupt image
    r = client.post(
        "/api/predict",
        files={"image": ("corrupt.jpg", b"not_an_image", "image/jpeg")},
        data={"model": "attention_fusion"}
    )
    assert r.status_code in [400, 415, 422]

    # 4. Grad-CAM on DeiT-Tiny must return 400 Bad Request
    r = client.post(
        "/api/predict/gradcam",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model": "deit_tiny"}
    )
    assert r.status_code == 400
    assert "pure Vision Transformer" in r.json()["detail"]

    # 5. Verify no local filesystem paths (e.g. D:\ or C:\) are exposed
    for resp in [r]:
        detail = resp.json().get("detail", "")
        assert "D:\\" not in detail
        assert "C:\\" not in detail
        assert "Desktop" not in detail
