# Potato Disease Research Demo

**Research Title:**  
*An Attention-Guided Lightweight CNN–ViT Fusion Network for Potato Leaf Disease Classification Across Controlled and Natural Environments*

---

## 1. Project Overview

This repository hosts the research demonstration web application for classifying potato leaf diseases across controlled laboratory conditions (e.g., PlantVillage) and complex natural field environments (e.g., PLDD-UP). The application demonstrates the empirical efficacy, generalization ability, attention mechanics, and explainability of four comparative architectures:

1. **Lightweight CNN Baseline:** EfficientNet-B0
2. **Compact Vision Transformer Baseline:** DeiT-Tiny
3. **Feature Concatenation Fusion:** CNN–ViT Concat Baseline
4. **Proposed Architecture:** Attention-Guided Lightweight CNN–ViT Fusion Network

### Key Features
- **Research Overview:** Interactive summary of research objectives, methodology, and model architectures.
- **Inference & Prediction:** Single and batch potato leaf image prediction across the four comparative models.
- **Explainability & Visualizations:**
  - Prediction probability distributions across disease classes.
  - Attention weight heatmaps / distributions for the attention-guided fusion mechanism.
  - Grad-CAM saliency maps for CNN convolutional feature activations.
- **Benchmarking & Evaluation:** Comprehensive model comparisons showing internal/external test metrics, generalization gap, parameter efficiency, and statistical significance tests from the research paper.

---

## 2. Technology Stack

### Frontend
- **Framework:** Next.js (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts & Visualizations:** Recharts
- **Icons:** Lucide React

### Backend
- **Framework:** FastAPI (Python 3.10+)
- **Deep Learning:** PyTorch, `timm` (PyTorch Image Models)
- **Image Processing:** Pillow (PIL), NumPy
- **Server:** Uvicorn

---

## 3. Repository Architecture & Folder Responsibilities

```text
potato-disease-research-demo/
│
├── frontend/                     # Next.js TypeScript Web Application
│   ├── app/                      # Next.js App Router (layout, pages, routes)
│   │   ├── predict/              # Model prediction & explainability page
│   │   └── results/              # Research metrics, comparison, & charts page
│   ├── components/               # Modular UI components
│   │   ├── layout/               # Navbar, Footer, Shell
│   │   ├── home/                 # Hero, Research Overview, Workflow
│   │   ├── prediction/           # Uploader, Selector, Cards, Charts, Grad-CAM
│   │   └── results/              # Metrics tables, comparison cards, charts
│   ├── data/                     # Static research findings & model definitions
│   ├── lib/                      # REST API client & UI utilities
│   ├── types/                    # TypeScript interfaces & API response types
│   ├── public/                   # Static assets (sample images, figures)
│   ├── package.json              # Frontend dependencies and scripts
│   ├── tsconfig.json             # TypeScript configuration
│   └── .env.example              # Frontend environment variables template
│
├── backend/                      # FastAPI Python Application
│   ├── app/                      # Application source code
│   │   ├── api/                  # API route handlers (/health, /models, /predict)
│   │   ├── models/               # PyTorch model definitions (4 architectures)
│   │   ├── services/             # Preprocessing, inference, Grad-CAM, attention
│   │   ├── schemas/              # Pydantic request & response schemas
│   │   └── core/                 # App configuration & settings
│   ├── configs/                  # Training & preprocessing configuration JSONs
│   ├── checkpoints/              # Directory for trained PyTorch model weights (.pt/.pth)
│   │   ├── efficientnet_b0/      # EfficientNet-B0 weights
│   │   ├── deit_tiny/            # DeiT-Tiny weights
│   │   ├── cnn_vit_concat/       # Concat Fusion weights
│   │   └── attention_fusion/     # Proposed Attention-Guided Fusion weights
│   ├── tests/                    # Backend unit and integration tests
│   ├── requirements.txt          # Python dependencies
│   └── .env.example              # Backend environment variables template
│
├── research/                     # Structured Paper Results & Reference Figures
│   ├── results/                  # Paper evaluation metrics in JSON format
│   └── figures/                  # Publication figures (curves, matrices, Grad-CAM)
│
├── docs/                         # Technical documentation & research specs
│   └── README.md
│
├── .gitignore                    # Git ignore configuration
├── README.md                     # Root project documentation
└── LICENSE                       # MIT License
```

---

## 4. Setup and Installation

### Prerequisites
- Node.js (v18.x or later) & npm
- Python 3.10+ & `pip`
- (Optional) CUDA-compatible GPU for accelerated PyTorch inference

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```
5. Place trained model checkpoints in their corresponding subdirectories under `backend/checkpoints/` (see [backend/checkpoints/README.md](backend/checkpoints/README.md)).
6. Launch the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   Interactive API documentation will be available at `http://localhost:8000/docs`.

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Copy the environment configuration:
   ```bash
   cp .env.example .env.local
   ```
4. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

---

## 5. Model Checkpoints & Data Isolation Notice

- **Datasets:** Raw datasets (PlantVillage, PLDD-UP), augmented datasets, and Jupyter training notebooks are kept outside this repository (stored in Google Drive).
- **Model Checkpoints:** Model weight files (`.pt`, `.pth`, `.ckpt`) are excluded by `.gitignore` to maintain a lightweight, clean repository. Place weight files in `backend/checkpoints/<model_name>/` as documented in [backend/checkpoints/README.md](backend/checkpoints/README.md).
