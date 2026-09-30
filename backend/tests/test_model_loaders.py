import pytest
import torch
from PIL import Image
from pathlib import Path

from app.models.efficientnet import load_efficientnet_b0
from app.models.deit import load_deit_tiny
from app.models.concat_fusion import load_concat_fusion
from app.models.attention_fusion import load_attention_fusion
from app.services.inference import inference_service, RESEARCH_CLASSES


CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def test_load_efficientnet_b0():
    ckpt_path = CHECKPOINTS_DIR / "efficientnet_b0" / "efficientnet_b0_best_seed42.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    model = load_efficientnet_b0(ckpt_path, device="cpu")
    assert not model.training, "Model should be in eval() mode"

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 3)


def test_load_deit_tiny():
    ckpt_path = CHECKPOINTS_DIR / "deit_tiny" / "deit_tiny_best_seed42.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    model = load_deit_tiny(ckpt_path, device="cpu")
    assert not model.training, "Model should be in eval() mode"

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 3)


def test_load_concat_fusion():
    ckpt_path = CHECKPOINTS_DIR / "cnn_vit_concat" / "cnn_vit_concat_best_seed42.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    model = load_concat_fusion(ckpt_path, device="cpu")
    assert not model.training, "Model should be in eval() mode"

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 3)


def test_load_attention_fusion():
    ckpt_path = CHECKPOINTS_DIR / "attention_fusion" / "attention_cnn_vit_best_seed42.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    model = load_attention_fusion(ckpt_path, device="cpu")
    assert not model.training, "Model should be in eval() mode"

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out, attn = model(dummy_input, return_attention=True)
    assert out.shape == (1, 3)
    assert attn.shape == (1, 2)
    assert torch.isclose(attn.sum(), torch.tensor(1.0), atol=1e-5)


def test_inference_service_loads_all_models():
    models = ["efficientnet_b0", "deit_tiny", "cnn_vit_concat", "attention_fusion"]
    for m_id in models:
        m = inference_service.get_or_load_model(m_id, seed=42)
        assert m is not None
        assert not m.training


def test_inference_service_prediction():
    test_img = Image.new("RGB", (256, 256), color=(34, 139, 34))

    for m_id in ["efficientnet_b0", "deit_tiny", "cnn_vit_concat", "attention_fusion"]:
        result = inference_service.predict(m_id, test_img, seed=42, explain=True)
        assert result.predicted_class in [c["name"] for c in RESEARCH_CLASSES]
        assert len(result.probabilities) == 3
        prob_sum = sum(p.probability for p in result.probabilities)
        assert 0.99 <= prob_sum <= 1.01
        assert result.inference_time_ms > 0

        if m_id == "attention_fusion":
            assert result.attention_weights_available is True
            assert result.attention_data is not None
            assert "cnn_branch_weight" in result.attention_data
            assert "vit_branch_weight" in result.attention_data
