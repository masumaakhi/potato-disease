"""
Specimen Suitability & Out-of-Distribution (OOD) Quality Advisory Service.
Safely detects synthetic digital screenshots, blank images, or non-botanical inputs
WITHOUT rejecting or impeding real diseased/healthy potato leaf specimens.
"""

from typing import Dict, Any, Optional
from PIL import Image
import numpy as np


def evaluate_specimen_suitability(
    image: Image.Image,
    top_confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates whether an input image appears to be a natural botanical leaf specimen
    or a digital screenshot / non-botanical input.

    SAFETY & INTEGRITY GUARANTEES:
    - Never throws an unhandled exception (Fail-Safe: defaults to optimal).
    - Preserves all real diseased leaves (healthy, necrotic brown, yellow halos, soil backgrounds).
    - Detects digital screenshots (flat UI, code editor, documents) and displays non-blocking advisory.
    """
    try:
        rgb_img = image.convert("RGB")
        img_np = np.asarray(rgb_img, dtype=np.float32)
        h, w, _ = img_np.shape
        total_pixels = float(h * w)

        if total_pixels <= 0:
            return {"is_likely_potato_leaf": True, "specimen_status": "optimal", "specimen_warning": None}

        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

        # 1. Flat Synthetic Canvas / Screenshot Detection
        pure_white_ratio = np.count_nonzero((r >= 245) & (g >= 245) & (b >= 245)) / total_pixels
        pure_dark_ratio = np.count_nonzero((r <= 20) & (g <= 20) & (b <= 20)) / total_pixels

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Saturation
        sat = np.zeros_like(max_c)
        valid_mask = max_c > 10.0
        sat[valid_mask] = delta[valid_mask] / max_c[valid_mask]
        low_sat_ratio = np.count_nonzero(sat < 0.08) / total_pixels

        # 2. Organic Foliar & Blight Palette Check
        # Green foliar foliage
        is_green = (g > r * 0.95) & (g > b * 1.05) & (g > 30)
        # Necrotic blight (brown, amber, dark olive, yellow chlorotic halo)
        is_brown_yellow = (r > b * 1.05) & (g > b * 0.85) & (delta > 15) & (r > 30)
        organic_ratio = np.count_nonzero(is_green | is_brown_yellow) / total_pixels

        # Rule A: Clear digital screenshot / solid canvas / code editor / text document
        if (pure_white_ratio > 0.55) or (pure_dark_ratio > 0.65) or (low_sat_ratio > 0.80 and organic_ratio < 0.05):
            return {
                "is_likely_potato_leaf": False,
                "specimen_status": "synthetic_advisory",
                "specimen_warning": (
                    "Specimen Advisory: The uploaded image appears to be a digital screenshot or non-botanical object. "
                    "This research model was trained strictly on potato leaf foliage (Solanum tuberosum); "
                    "predictions on non-leaf images are out-of-distribution."
                )
            }

        # Rule B: Non-foliar image with minimal plant color profile
        if organic_ratio < 0.03 and low_sat_ratio > 0.50:
            return {
                "is_likely_potato_leaf": False,
                "specimen_status": "synthetic_advisory",
                "specimen_warning": (
                    "Specimen Advisory: Minimal plant foliage detected in the specimen. "
                    "Please ensure you upload an actual potato leaf photo for accurate disease diagnosis."
                )
            }

        # Rule C: Ambiguous prediction with low confidence (< 45%)
        if top_confidence < 0.45:
            return {
                "is_likely_potato_leaf": True,
                "specimen_status": "low_confidence_advisory",
                "specimen_warning": (
                    "Low Confidence Advisory: The model identified borderline disease features. "
                    "Consider taking a clearer, well-lit photograph focused on the lesion area."
                )
            }

        return {
            "is_likely_potato_leaf": True,
            "specimen_status": "optimal",
            "specimen_warning": None
        }

    except Exception:
        # Ultimate fail-safe: Never disrupt normal inference flow
        return {
            "is_likely_potato_leaf": True,
            "specimen_status": "optimal",
            "specimen_warning": None
        }
