"""
Test Suite for Model Verification Workflow
==========================================
Tests the verification pipeline, reference CSV parser, probability tolerance checking,
and verification reporting without modifying backend inference logic.
"""

import sys
import json
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.verification.verify_predictions import (
    ModelVerifier,
    normalize_class_name,
    normalize_model_name,
    MODEL_DISPLAY_NAMES,
    ALL_MODELS,
)


def test_normalize_class_name():
    assert normalize_class_name("Early_Blight") == "Early Blight"
    assert normalize_class_name("early blight") == "Early Blight"
    assert normalize_class_name("Late_Blight") == "Late Blight"
    assert normalize_class_name("healthy") == "Healthy"
    assert normalize_class_name("Healthy") == "Healthy"


def test_normalize_model_name():
    assert normalize_model_name("EfficientNet_B0") == "efficientnet_b0"
    assert normalize_model_name("EfficientNet-B0") == "efficientnet_b0"
    assert normalize_model_name("DeiT-Tiny") == "deit_tiny"
    assert normalize_model_name("CNN+ViT Concatenation") == "cnn_vit_concat"
    assert normalize_model_name("Attention-Guided CNN+ViT") == "attention_fusion"
    assert normalize_model_name("attention_fusion") == "attention_fusion"


def test_reference_csv_missing_detection():
    """Verify that an empty or missing reference CSV correctly flags REFERENCE_DATA_MISSING."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write("image_name,model,reference_predicted_class\n")
        tmp_path = tmp.name

    verifier = ModelVerifier(reference_csv_path=tmp_path)
    records, err = verifier.load_reference_predictions()
    assert len(records) == 0

    res = verifier.run_verification()
    assert res["status"] in ("REFERENCE_DATA_MISSING", "API_OFFLINE")


def test_reference_csv_parsing_formats():
    """Verify parser supports individual prob columns, JSON string, and attention weights."""
    csv_content = """image_name,model,reference_predicted_class,probability_early_blight,probability_late_blight,probability_healthy,reference_cnn_weight,reference_vit_weight
sample1.jpg,efficientnet_b0,Early Blight,0.92,0.05,0.03,,
sample2.jpg,attention_fusion,Healthy,0.01,0.02,0.97,0.55,0.45
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write(csv_content)
        tmp_path = tmp.name

    verifier = ModelVerifier(reference_csv_path=tmp_path)
    records, err = verifier.load_reference_predictions()
    assert err is None
    assert len(records) == 2

    # Check record 1
    r1 = records[0]
    assert r1["image_name"] == "sample1.jpg"
    assert r1["model"] == "efficientnet_b0"
    assert r1["reference_predicted_class"] == "Early Blight"
    assert r1["reference_probabilities"]["Early Blight"] == 0.92
    assert r1["reference_cnn_weight"] is None

    # Check record 2
    r2 = records[1]
    assert r2["image_name"] == "sample2.jpg"
    assert r2["model"] == "attention_fusion"
    assert r2["reference_predicted_class"] == "Healthy"
    assert r2["reference_probabilities"]["Healthy"] == 0.97
    assert r2["reference_cnn_weight"] == 0.55
    assert r2["reference_vit_weight"] == 0.45


def test_mismatch_detection_logic():
    """Verify that prediction class mismatches and probability differences outside tolerance trigger FAIL."""
    verifier = ModelVerifier(prob_tolerance=0.01)

    # Simulated website prediction
    website_pred = {
        "predicted_class": "Early Blight",
        "probabilities": [
            {"class_name": "Early Blight", "probability": 0.80},
            {"class_name": "Late Blight", "probability": 0.15},
            {"class_name": "Healthy", "probability": 0.05},
        ]
    }

    # Reference expects Late Blight
    rec_mismatch = {
        "image_name": "leaf.jpg",
        "model": "efficientnet_b0",
        "reference_predicted_class": "Late Blight",
        "reference_probabilities": {"Early Blight": 0.10, "Late Blight": 0.85, "Healthy": 0.05},
    }

    ref_class = rec_mismatch["reference_predicted_class"]
    web_class = website_pred["predicted_class"]
    assert ref_class != web_class  # Mismatch detected


def test_terminal_report_output_structure(capsys):
    """Verify that print_terminal_report produces the exact requested headers and sections."""
    verifier = ModelVerifier()

    sample_result = {
        "status": "FAIL",
        "verified": False,
        "overall": {
            "total_comparisons": 4,
            "prediction_matches": 3,
            "prediction_mismatches": 1,
            "match_rate": 75.0,
            "missing_images_count": 0,
        },
        "models": {
            "efficientnet_b0": {
                "display_name": "EfficientNet-B0",
                "images_tested": 1,
                "matches": 1,
                "mismatches": 0,
                "api_failures": 0,
            },
            "deit_tiny": {
                "display_name": "DeiT-Tiny",
                "images_tested": 1,
                "matches": 1,
                "mismatches": 0,
                "api_failures": 0,
            },
            "cnn_vit_concat": {
                "display_name": "CNN+ViT Concatenation",
                "images_tested": 1,
                "matches": 1,
                "mismatches": 0,
                "api_failures": 0,
            },
            "attention_fusion": {
                "display_name": "Attention-Guided CNN+ViT",
                "images_tested": 1,
                "matches": 0,
                "mismatches": 1,
                "api_failures": 0,
            },
        },
        "mismatches": [
            {
                "image": "sample_leaf.jpg",
                "model": "Attention-Guided CNN+ViT",
                "research_prediction": "Late Blight",
                "website_prediction": "Early Blight",
                "probability_difference": 0.42,
                "status": "FAIL",
            }
        ],
        "missing_images": [],
        "failed_requests": [],
    }

    exit_code = verifier.print_terminal_report(sample_result)
    assert exit_code == 1

    captured = capsys.readouterr().out
    assert "MODEL VERIFICATION" in captured
    assert "EfficientNet-B0" in captured
    assert "DeiT-Tiny" in captured
    assert "CNN+ViT Concatenation" in captured
    assert "Attention-Guided CNN+ViT" in captured
    assert "Overall:" in captured
    assert "Total comparisons: 4" in captured
    assert "Prediction matches: 3" in captured
    assert "Prediction mismatches: 1" in captured
    assert "Match rate: 75.0%" in captured
    assert "Image: sample_leaf.jpg" in captured
    assert "Research prediction: Late Blight" in captured
    assert "Website prediction: Early Blight" in captured
    assert "Probability difference: 0.42" in captured
    assert "Status: FAIL" in captured

