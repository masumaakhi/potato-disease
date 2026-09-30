---
title: Potato Disease AI Backend
emoji: 🥔
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# Backend Service - Potato Disease Research Demo

FastAPI-powered inference service for potato leaf disease classification and model interpretability.

## Directory Layout

```text
backend/
├── app/
│   ├── main.py              # FastAPI application entrypoint & middleware
│   ├── api/                 # REST API endpoints (health, models, predict)
│   ├── models/              # PyTorch model definitions for the 4 architectures
│   ├── services/            # Preprocessing, inference engine, Grad-CAM, attention
│   ├── schemas/             # Pydantic data validation schemas
│   └── core/                # Configuration and environment variables
├── configs/                 # Model and preprocessing JSON configuration files
├── checkpoints/             # Trained PyTorch model weights (.pt / .pth)
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variables template
```

## Model Checkpoint Placement Guide

Trained model weights should be placed into the respective subdirectories inside `backend/checkpoints/`:

1. **EfficientNet-B0 (CNN Baseline):**
   - Directory: `checkpoints/efficientnet_b0/`
   - Target weights: e.g., `efficientnet_b0_best.pth`
2. **DeiT-Tiny (ViT Baseline):**
   - Directory: `checkpoints/deit_tiny/`
   - Target weights: e.g., `deit_tiny_best.pth`
3. **CNN–ViT Concat (Feature Fusion Baseline):**
   - Directory: `checkpoints/cnn_vit_concat/`
   - Target weights: e.g., `cnn_vit_concat_best.pth`
4. **Attention-Guided CNN–ViT Fusion (Proposed Architecture):**
   - Directory: `checkpoints/attention_fusion/`
   - Target weights: e.g., `attention_fusion_best.pth`

> Note: All checkpoint binary files (`.pt`, `.pth`, `.ckpt`) are excluded by `.gitignore` to keep the codebase clean. Ensure files are copied locally prior to starting the production inference service.

## Configuration Files

The JSON files located in `backend/configs/` specify architecture parameters, input dimensions, normalization statistics, and seed configurations. These files will be populated from your research Google Drive.
