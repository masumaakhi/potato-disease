# Potato Disease Research — Model Verification Suite

## 1. Overview & Verification Philosophy

This verification suite rigorously validates that the **FastAPI backend inference layer** produces predictions, probability distributions, and attention weights that are **empirically and numerically consistent** with the original **Google Colab research pipeline** (`Complete_Potato_CNN_ViT_Research_Pipeline.ipynb`).

> [!IMPORTANT]
> **A successful HTTP 200 response is NOT model verification.**
> Verification requires establishing numerical consistency with the original research pipeline across all four architectures:
> 1. **EfficientNet-B0** (Lightweight CNN Baseline)
> 2. **DeiT-Tiny** (Compact Vision Transformer Baseline)
> 3. **CNN+ViT Concatenation** (Direct Feature Concatenation)
> 4. **Attention-Guided CNN+ViT** (Proposed Cross-Attention Network)

> [!WARNING]
> **Strict Anti-Fabrication Policy:**
> Reference predictions must **never** be fabricated, faked, or generated via random numbers. If reference predictions are not yet loaded, the system explicitly reports `REFERENCE_DATA_MISSING` and will **not** claim the models are verified until real research comparison actually passes.

---

## 2. Directory Structure

```text
research/
└── verification/
    ├── README.md                                 # Complete verification guide (this file)
    ├── reference_predictions.csv                 # Target reference prediction file
    ├── verification_report.json                  # Machine-readable verification output
    ├── verify_predictions.py                     # Automated verification runner
    ├── export_reference_predictions_colab.py     # Colab export script
    └── images/                                   # Folder for exported test images
```

---

## 3. Workflow Guide

### Step A: Export Reference Predictions from the Original Research Pipeline

Run this inside your **Google Colab** environment (where the research notebook and full test sets reside):

#### Option 1: Using the provided export script
```bash
python research/verification/export_reference_predictions_colab.py \
    --project_root "/content/drive/MyDrive/Potato_CNN_ViT_Research" \
    --split external_test \
    --sample_count 20 \
    --output_dir "./verification_export"
```

#### Option 2: Running directly in a Google Colab notebook cell
Paste the following snippet into a Colab cell after loading your checkpoints:

```python
import pandas as pd
import json
import torch
from pathlib import Path
from PIL import Image

# 1. Load external test split
split_df = pd.read_csv("/content/drive/MyDrive/Potato_CNN_ViT_Research/03_splits/external_test.csv")
sample_df = split_df.groupby("unified_class").head(5).reset_index(drop=True)

export_rows = []
images_dir = Path("./verification_export/images")
images_dir.mkdir(parents=True, exist_ok=True)

# 2. Iterate through models and test images
for model_key in ["efficientnet_b0", "deit_tiny", "cnn_vit_concat", "attention_fusion"]:
    # model = load_checkpoint_for_model(model_key)
    # model.eval()
    for _, row in sample_df.iterrows():
        img_path = Path(row["path"])
        dest_img = images_dir / img_path.name
        if not dest_img.exists():
            import shutil
            shutil.copy2(img_path, dest_img)

        # with torch.no_grad():
        #     output = model(transform(Image.open(img_path).convert('RGB')).unsqueeze(0))
        #     probs = torch.softmax(output, dim=1).squeeze().tolist()
        #     pred_class = ["Early Blight", "Late Blight", "Healthy"][torch.argmax(output).item()]
        
        # export_rows.append({
        #     "image_name": img_path.name,
        #     "model": model_key,
        #     "reference_predicted_class": pred_class,
        #     "probability_early_blight": probs[0],
        #     "probability_late_blight": probs[1],
        #     "probability_healthy": probs[2],
        #     "reference_cnn_weight": cnn_w if model_key == "attention_fusion" else None,
        #     "reference_vit_weight": vit_w if model_key == "attention_fusion" else None,
        # })

# 3. Save reference CSV
# pd.DataFrame(export_rows).to_csv("./verification_export/reference_predictions.csv", index=False)
```

---

### Step B: Place the Reference CSV & Images in the Project

1. Copy `reference_predictions.csv` to:
   ```text
   research/verification/reference_predictions.csv
   ```
2. Copy the exported test images to:
   ```text
   research/verification/images/
   ```
   *(Note: The verification tool also automatically searches `frontend/public/samples/` if benchmark images are located there).*

#### Expected Reference CSV Schema:
| Column | Description | Example |
| :--- | :--- | :--- |
| `image_name` | Filename of test image | `sample_early_blight.jpg` |
| `model` | Target architecture identifier | `attention_fusion` or `EfficientNet-B0` |
| `reference_predicted_class` | Ground-truth research prediction | `Early Blight` |
| `probability_early_blight` | Softmax probability for Early Blight | `0.9421` |
| `probability_late_blight` | Softmax probability for Late Blight | `0.0412` |
| `probability_healthy` | Softmax probability for Healthy | `0.0167` |
| `reference_cnn_weight` | *(Optional)* CNN attention weight (Attention model) | `0.5481` |
| `reference_vit_weight` | *(Optional)* ViT attention weight (Attention model) | `0.4519` |

---

### Step C: Start the FastAPI Backend Server

Open a terminal and start the FastAPI service:
```powershell
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Verify the backend is live:
```bash
curl http://127.0.0.1:8000/api/health
# Returns: {"status":"healthy","service":"Potato Disease Research Demo","checkpoints_available":true}
```

---

### Step D: Run the Verification Script

Run the automated verification suite from the project root:
```powershell
python research/verification/verify_predictions.py
```

#### Optional Command-Line Arguments:
```powershell
python research/verification/verify_predictions.py \
    --api-url "http://127.0.0.1:8000" \
    --reference-csv "research/verification/reference_predictions.csv" \
    --images-dir "research/verification/images" \
    --prob-tolerance 0.005 \
    --attn-tolerance 0.005 \
    --report-json "research/verification/verification_report.json"
```

#### Automated Unit & Verification Testing (Pytest):
```powershell
python -m pytest backend/tests/test_model_verification.py -v
```

---

### Step E: Interpret PASS / FAIL Results

The runner outputs a structured terminal diagnostic table:

```text
MODEL VERIFICATION
==================

EfficientNet-B0
Images tested: 20
Prediction matches: 20/20
Prediction mismatches: 0
API failures: 0

DeiT-Tiny
Images tested: 20
Prediction matches: 20/20
Prediction mismatches: 0
API failures: 0

CNN+ViT Concatenation
Images tested: 20
Prediction matches: 20/20
Prediction mismatches: 0
API failures: 0

Attention-Guided CNN+ViT
Images tested: 20
Prediction matches: 20/20
Prediction mismatches: 0
API failures: 0

Overall:
Total comparisons: 80
Prediction matches: 80
Prediction mismatches: 0
Match rate: 100.0%

==================================================
FINAL VERDICT: VERIFIED (PASS)
Report saved to: research/verification/verification_report.json
==================================================
```

#### If a Mismatch Occurs:
The tool highlights the exact error details:
```text
DETAILED MISMATCHES:
--------------------
Image: specimen_042.jpg
Model: Attention-Guided CNN+ViT
Research prediction: Late Blight
Website prediction: Early Blight
Probability difference: 0.3841
Status: FAIL
```

#### Numerical Tolerances Explained:
- **`--prob-tolerance 0.005` (0.5%)**: Accounts for standard floating-point precision differences between CUDA (GPU) and PyTorch CPU inference implementations.
- **`--attn-tolerance 0.005`**: Maximum allowable variance for dynamic branch attention gating weights.

---

## 4. Frontend Direct-Pass Verification

The frontend has been verified to adhere strictly to scientific transparency:
1. **Zero Client Calculation:** [PredictPage](file:///d:/Desktop/Potato%20Disease/frontend/app/predict/page.tsx) directly consumes the JSON payload from `POST /api/predict`.
2. **Untouched Probabilities:** [ProbabilityChart](file:///d:/Desktop/Potato%20Disease/frontend/components/prediction/probability-chart.tsx) renders the server-provided softmax distribution directly.
3. **Authentic Attention Weights:** [AttentionWeights](file:///d:/Desktop/Potato%20Disease/frontend/components/prediction/attention-weights.tsx) maps `cnn_branch_weight` and `vit_branch_weight` exactly as output by the PyTorch model without any client-side bias or rounding alterations.
