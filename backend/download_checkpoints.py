"""
Optional Helper Script: Checkpoint Downloader for Production Hosting
Downloads the 4 required research model checkpoints if not already present.
"""

import os
from pathlib import Path
import urllib.request
import sys

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Mapping of model ID to relative directory and default filename
MODEL_FILES = {
    "efficientnet_b0": ("efficientnet_b0", "efficientnet_b0_best_seed42.pth"),
    "deit_tiny": ("deit_tiny", "deit_tiny_best_seed42.pth"),
    "cnn_vit_concat": ("cnn_vit_concat", "cnn_vit_concat_best_seed42.pth"),
    "attention_fusion": ("attention_fusion", "attention_cnn_vit_best_seed42.pth"),
}

def verify_checkpoints() -> bool:
    """Verifies whether all 4 model checkpoints are available locally."""
    missing = []
    for model_id, (subdir, filename) in MODEL_FILES.items():
        target_path = CHECKPOINTS_DIR / subdir / filename
        # Also check for any .pth file in that directory
        folder = CHECKPOINTS_DIR / subdir
        pth_files = list(folder.glob("*.pth")) if folder.exists() else []
        if not target_path.exists() and not pth_files:
            missing.append(f"{subdir}/{filename}")
    
    if missing:
        print(f"[WARNING] Missing checkpoints: {missing}")
        return False
    print("[SUCCESS] All required model checkpoints are present locally.")
    return True

if __name__ == "__main__":
    verify_checkpoints()
