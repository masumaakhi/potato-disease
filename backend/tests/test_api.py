import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(40, 140, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoints():
    # Test root /health
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"

    # Test api /api/health
    res_api = client.get("/api/health")
    assert res_api.status_code == 200
    assert res_api.json()["status"] == "healthy"


def test_get_models_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 4

    model_names = [m["name"] for m in models]
    expected_names = [
        "EfficientNet-B0",
        "DeiT-Tiny",
        "CNN–ViT Concatenation",
        "Attention-Guided CNN–ViT"
    ]
    for expected in expected_names:
        assert expected in model_names

    # Checkpoint availability should now be True as real checkpoints are loaded
    for m in models:
        assert m["checkpoint_available"] is True


def test_get_individual_model():
    response = client.get("/api/models/attention_fusion")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "attention_fusion"
    assert data["name"] == "Attention-Guided CNN–ViT"


def test_predict_status():
    response = client.get("/api/predict/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "preprocessing" in data
    assert data["preprocessing"]["resize"] == [256, 256]
    assert data["preprocessing"]["center_crop"] == 224


def test_cors_headers():
    response = client.options(
        "/api/models",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_predict_endpoint_valid_image():
    img_bytes = create_jpeg_bytes()
    response = client.post(
        "/api/predict",
        data={"model_id": "efficientnet_b0", "explain": "false"},
        files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "efficientnet_b0"
    assert data["predicted_class"] in ["Early Blight", "Late Blight", "Healthy"]
    assert len(data["probabilities"]) == 3
    assert data["confidence"] > 0


def test_predict_endpoint_corrupt_file():
    corrupt_bytes = b"Not a real image file content."
    response = client.post(
        "/api/predict",
        data={"model_id": "efficientnet_b0", "explain": "false"},
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    )
    assert response.status_code in [400, 415, 422]
