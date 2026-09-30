import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def get_test_image_bytes(format="JPEG") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(45, 130, 60))
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.mark.parametrize("model_id,expected_name", [
    ("efficientnet_b0", "EfficientNet-B0"),
    ("deit_tiny", "DeiT-Tiny"),
    ("cnn_vit_concat", "CNN–ViT Concatenation"),
    ("attention_fusion", "Attention-Guided CNN–ViT"),
])
def test_predict_all_four_models_with_image_and_model_fields(model_id, expected_name):
    """Test POST /api/predict using the exact fields requested: 'image' and 'model'."""
    img_bytes = get_test_image_bytes()

    response = client.post(
        "/api/predict",
        data={"model": model_id},
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200, f"Failed for {model_id}: {response.text}"
    data = response.json()

    # Required fields
    assert data["model_id"] == model_id
    assert data["model_name"] == expected_name
    assert data["predicted_class"] in ["Early Blight", "Late Blight", "Healthy"]
    assert 0.0 <= data["confidence"] <= 1.0

    # Probabilities
    probs = data["probabilities"]
    assert len(probs) == 3
    prob_sum = sum(p["probability"] for p in probs)
    assert 0.99 <= prob_sum <= 1.01

    # Inference info
    assert "inference_info" in data
    assert data["inference_info"] is not None
    assert data["inference_info"]["inference_time_ms"] > 0
    assert data["inference_info"]["input_resolution"] == "224x224"
    assert "timestamp" in data["inference_info"]

    # Special check for attention-fusion model
    if model_id == "attention_fusion":
        assert data["attention_weights"] is not None
        attn = data["attention_weights"]
        assert "cnn_branch_weight" in attn
        assert "vit_branch_weight" in attn
        assert "dominant_branch" in attn
        assert attn["dominant_branch"] in ["CNN", "ViT"]
        assert 0.0 <= attn["cnn_branch_weight"] <= 1.0
        assert 0.0 <= attn["vit_branch_weight"] <= 1.0
        assert abs(attn["cnn_branch_weight"] + attn["vit_branch_weight"] - 1.0) < 0.01


def test_predict_alias_fields():
    """Verify alias support: 'file' and 'model_id' work interchangeably with 'image' and 'model'."""
    img_bytes = get_test_image_bytes(format="PNG")

    response = client.post(
        "/api/predict",
        data={"model_id": "attention_fusion"},
        files={"file": ("leaf.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "attention_fusion"
    assert data["attention_weights"] is not None


def test_predict_missing_image():
    """Missing image upload returns 400 Bad Request."""
    response = client.post(
        "/api/predict",
        data={"model": "efficientnet_b0"}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_missing_model():
    """Missing model parameter returns 400 Bad Request."""
    img_bytes = get_test_image_bytes()
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_predict_unsupported_model():
    """Unsupported model name returns 400 with list of supported models."""
    img_bytes = get_test_image_bytes()
    response = client.post(
        "/api/predict",
        data={"model": "resnet50"},
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"].lower()


def test_predict_corrupt_image():
    """Corrupt image returns appropriate client error without internal paths."""
    corrupt_data = b"GIF89a\x00\x00\x00\x00\xff\xff"
    response = client.post(
        "/api/predict",
        data={"model": "efficientnet_b0"},
        files={"image": ("corrupt.jpg", corrupt_data, "image/jpeg")}
    )
    assert response.status_code in [400, 415, 422]
    # Ensure no internal filesystem paths are leaked
    body_lower = response.text.lower()
    assert "checkpoints" not in body_lower
    assert "c:\\" not in body_lower
    assert "d:\\" not in body_lower


def test_predict_empty_file():
    """Empty 0-byte file returns 400 Bad Request."""
    response = client.post(
        "/api/predict",
        data={"model": "efficientnet_b0"},
        files={"image": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
