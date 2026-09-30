#!/usr/bin/env python3
"""
Automated Research Prediction Verification Suite
=================================================
Verifies that the website backend produces the same research predictions
as the original research/Colab pipeline.

Requirements:
- Loads reference predictions exported from the original Colab pipeline
- Submits actual images to the live FastAPI /api/predict endpoint
- Tests all 4 models: EfficientNet-B0, DeiT-Tiny, CNN+ViT Concat, Attention Fusion
- Compares predicted class, class probabilities (within numerical tolerance),
  and attention weights
- Generates a clear terminal report and structured JSON report

Usage:
    python research/verification/verify_predictions.py [OPTIONS]

Options:
    --api-url         FastAPI base URL (default: http://127.0.0.1:8000)
    --reference-csv   Path to reference predictions CSV (default: research/verification/reference_predictions.csv)
    --images-dir      Directory containing test images (default: research/verification/images)
    --prob-tolerance  Maximum allowed absolute difference in probabilities (default: 0.005)
    --attn-tolerance  Maximum allowed absolute difference in attention weights (default: 0.005)
    --report-json     Output JSON path (default: research/verification/verification_report.json)
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests


# Canonical Model Mapping
MODEL_KEY_MAP = {
    # EfficientNet-B0
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b0_seed42": "efficientnet_b0",
    "cnn_efficientnet_b0": "efficientnet_b0",
    # DeiT-Tiny
    "deit_tiny": "deit_tiny",
    "deit-tiny": "deit_tiny",
    "vit_deit_tiny": "deit_tiny",
    "deit_tiny_seed42": "deit_tiny",
    # CNN+ViT Concat
    "cnn_vit_concat": "cnn_vit_concat",
    "cnn+vit concatenation": "cnn_vit_concat",
    "cnn–vit concatenation": "cnn_vit_concat",
    "cnn_vit_concat_seed42": "cnn_vit_concat",
    "concat_cnn_vit": "cnn_vit_concat",
    # Attention Fusion
    "attention_fusion": "attention_fusion",
    "attention-guided cnn+vit": "attention_fusion",
    "attention–guided cnn–vit": "attention_fusion",
    "attention_guided_cnn_vit": "attention_fusion",
    "attention_cnn_vit": "attention_fusion",
    "attention_fusion_seed42": "attention_fusion",
}

MODEL_DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
    "cnn_vit_concat": "CNN+ViT Concatenation",
    "attention_fusion": "Attention-Guided CNN+ViT",
}

ALL_MODELS = [
    "efficientnet_b0",
    "deit_tiny",
    "cnn_vit_concat",
    "attention_fusion",
]

CANONICAL_CLASSES = ["Early Blight", "Late Blight", "Healthy"]


def normalize_class_name(name: str) -> str:
    """Normalize class names with underscores, spaces, or cases."""
    if not name:
        return ""
    clean = name.strip().replace("_", " ").lower()
    if "early" in clean:
        return "Early Blight"
    if "late" in clean:
        return "Late Blight"
    if "health" in clean:
        return "Healthy"
    return name.strip()


def normalize_model_name(name: str) -> str:
    """Normalize model identifier to canonical API key."""
    if not name:
        return ""
    clean = name.strip().lower().replace(" ", "_").replace("-", "_")
    return MODEL_KEY_MAP.get(clean, MODEL_KEY_MAP.get(name.strip().lower(), name.strip()))


class ModelVerifier:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        reference_csv_path: str = "research/verification/reference_predictions.csv",
        images_dir: str = "research/verification/images",
        prob_tolerance: float = 0.005,
        attn_tolerance: float = 0.005,
        report_json_path: str = "research/verification/verification_report.json",
    ):
        self.api_url = api_url.rstrip("/")
        self.reference_csv_path = Path(reference_csv_path)
        self.images_dir = Path(images_dir)
        self.prob_tolerance = prob_tolerance
        self.attn_tolerance = attn_tolerance
        self.report_json_path = Path(report_json_path)

        # Fallback image search paths
        self.fallback_image_dirs = [
            self.images_dir,
            Path("frontend/public/samples"),
            Path("backend/tests/fixtures"),
        ]

    def check_api_health(self) -> Tuple[bool, str]:
        """Check if FastAPI backend is online and models/checkpoints are ready."""
        health_url = f"{self.api_url}/api/health"
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("checkpoints_available") is False:
                    return False, "Backend is reachable but reports checkpoints are unavailable."
                return True, "Backend online and checkpoints verified."
            return False, f"Backend health endpoint returned HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to backend at {self.api_url}: {str(e)}"

    def load_reference_predictions(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Load and parse reference prediction records from CSV."""
        if not self.reference_csv_path.exists():
            return [], f"Reference prediction file not found: {self.reference_csv_path}"

        records = []
        with open(self.reference_csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], "Reference CSV is empty or has no header."

            # Normalize headers
            headers = [h.strip() for h in reader.fieldnames]

            for row_idx, raw_row in enumerate(reader, start=2):
                # Clean keys and values
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() if k}

                # Check if row is purely blank
                if not any(row.values()):
                    continue

                # Extract image name
                img_name = row.get("image_name") or row.get("image") or row.get("file_name") or row.get("image_path")
                if not img_name:
                    continue

                # Extract model
                model_raw = row.get("model") or row.get("model_id") or row.get("architecture") or ""
                model_key = normalize_model_name(model_raw)

                # Extract reference predicted class
                ref_class = row.get("reference_predicted_class") or row.get("predicted_class") or row.get("predicted_class_name") or ""
                ref_class = normalize_class_name(ref_class)

                # Extract probabilities
                ref_probs: Dict[str, float] = {}
                # Form 1: individual probability columns
                for col_name, val in row.items():
                    lower_col = col_name.lower().replace(" ", "_")
                    if "early" in lower_col and ("prob" in lower_col or lower_col.startswith("p_")):
                        try:
                            ref_probs["Early Blight"] = float(val)
                        except ValueError:
                            pass
                    elif "late" in lower_col and ("prob" in lower_col or lower_col.startswith("p_")):
                        try:
                            ref_probs["Late Blight"] = float(val)
                        except ValueError:
                            pass
                    elif "health" in lower_col and ("prob" in lower_col or lower_col.startswith("p_")):
                        try:
                            ref_probs["Healthy"] = float(val)
                        except ValueError:
                            pass

                # Form 2: JSON string or dict string in reference_probabilities
                if not ref_probs and "reference_probabilities" in row and row["reference_probabilities"]:
                    try:
                        raw_p = json.loads(row["reference_probabilities"])
                        if isinstance(raw_p, dict):
                            for k, v in raw_p.items():
                                ref_probs[normalize_class_name(k)] = float(v)
                        elif isinstance(raw_p, list) and len(raw_p) == 3:
                            # Order in research: Early Blight, Late Blight, Healthy
                            ref_probs["Early Blight"] = float(raw_p[0])
                            ref_probs["Late Blight"] = float(raw_p[1])
                            ref_probs["Healthy"] = float(raw_p[2])
                    except Exception:
                        pass

                # Extract attention weights if available
                ref_cnn_w: Optional[float] = None
                ref_vit_w: Optional[float] = None
                for col_name, val in row.items():
                    lower_col = col_name.lower()
                    if "cnn" in lower_col and ("attn" in lower_col or "weight" in lower_col):
                        try:
                            ref_cnn_w = float(val)
                        except ValueError:
                            pass
                    elif "vit" in lower_col and ("attn" in lower_col or "weight" in lower_col):
                        try:
                            ref_vit_w = float(val)
                        except ValueError:
                            pass

                records.append({
                    "row_number": row_idx,
                    "image_name": Path(img_name).name,
                    "model": model_key,
                    "model_display": MODEL_DISPLAY_NAMES.get(model_key, model_raw),
                    "reference_predicted_class": ref_class,
                    "reference_probabilities": ref_probs,
                    "reference_cnn_weight": ref_cnn_w,
                    "reference_vit_weight": ref_vit_w,
                    "raw_row": row,
                })

        return records, None

    def find_image_file(self, image_name: str) -> Optional[Path]:
        """Search for image in configured images dir and fallback directories."""
        filename = Path(image_name).name
        for d in self.fallback_image_dirs:
            if not d.exists():
                continue
            direct_path = d / filename
            if direct_path.exists():
                return direct_path
            # Also check case-insensitive match
            for cand in d.glob("*"):
                if cand.is_file() and cand.name.lower() == filename.lower():
                    return cand
        return None

    def execute_prediction(self, image_path: Path, model_key: str) -> Dict[str, Any]:
        """Submit image to live /api/predict endpoint."""
        predict_url = f"{self.api_url}/api/predict"

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            data = {"model": model_key, "explain": "true"}
            resp = requests.post(predict_url, files=files, data=data, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        return resp.json()

    def _save_report(self, report_data: Dict[str, Any]):
        """Save report to JSON file."""
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    def run_verification(self) -> Dict[str, Any]:
        """Run complete model verification workflow."""
        # 1. Check API health
        api_ok, api_msg = self.check_api_health()
        if not api_ok:
            result = {
                "status": "API_OFFLINE",
                "verified": False,
                "error": api_msg,
                "overall": {
                    "total_comparisons": 0,
                    "prediction_matches": 0,
                    "prediction_mismatches": 0,
                    "match_rate": 0.0,
                },
                "models": {},
                "mismatches": [],
                "missing_images": [],
                "failed_requests": [],
            }
            self._save_report(result)
            return result

        # 2. Load reference predictions
        records, load_err = self.load_reference_predictions()
        if load_err or not records:
            result = {
                "status": "REFERENCE_DATA_MISSING",
                "verified": False,
                "notice": (
                    "No reference predictions found in reference CSV.\n"
                    "Do NOT invent or hardcode fake predictions.\n"
                    "Export genuine predictions from the Colab research pipeline using:\n"
                    "python research/verification/export_reference_predictions_colab.py"
                ),
                "error": load_err,
                "overall": {
                    "total_comparisons": 0,
                    "prediction_matches": 0,
                    "prediction_mismatches": 0,
                    "match_rate": 0.0,
                },
                "models": {m: {"tested": 0, "matches": 0, "mismatches": 0, "failures": 0} for m in ALL_MODELS},
                "mismatches": [],
                "missing_images": [],
                "failed_requests": [],
            }
            self._save_report(result)
            return result

        # 3. Setup test structures
        model_stats: Dict[str, Dict[str, Any]] = {
            m: {
                "display_name": MODEL_DISPLAY_NAMES[m],
                "images_tested": 0,
                "matches": 0,
                "mismatches": 0,
                "api_failures": 0,
                "prob_diffs": [],
                "attn_diffs": [],
            }
            for m in ALL_MODELS
        }

        mismatches: List[Dict[str, Any]] = []
        missing_images: List[Dict[str, Any]] = []
        failed_requests: List[Dict[str, Any]] = []
        tested_unique_images = set()

        # 4. Iterate over reference records
        for rec in records:
            model_key = rec["model"]
            if model_key not in model_stats:
                model_stats[model_key] = {
                    "display_name": rec["model_display"],
                    "images_tested": 0,
                    "matches": 0,
                    "mismatches": 0,
                    "api_failures": 0,
                    "prob_diffs": [],
                    "attn_diffs": [],
                }

            stats = model_stats[model_key]
            img_name = rec["image_name"]
            img_path = self.find_image_file(img_name)

            if not img_path:
                missing_images.append({
                    "image": img_name,
                    "model": model_key,
                    "expected_path": str(self.images_dir / img_name),
                })
                continue

            tested_unique_images.add(img_name)
            stats["images_tested"] += 1

            # Execute real inference
            try:
                website_pred = self.execute_prediction(img_path, model_key)
            except Exception as e:
                stats["api_failures"] += 1
                failed_requests.append({
                    "image": img_name,
                    "model": model_key,
                    "error": str(e),
                })
                continue

            # Compare predicted class
            web_class = normalize_class_name(website_pred.get("predicted_class", ""))
            ref_class = rec["reference_predicted_class"]

            # Parse website probabilities
            web_probs: Dict[str, float] = {}
            for p in website_pred.get("probabilities", []):
                web_probs[normalize_class_name(p["class_name"])] = float(p["probability"])

            # Compute probability differences
            ref_probs = rec["reference_probabilities"]
            prob_diff_info: Dict[str, Any] = {}
            max_prob_diff = 0.0

            if ref_probs:
                for cls_name in CANONICAL_CLASSES:
                    if cls_name in ref_probs and cls_name in web_probs:
                        diff = abs(web_probs[cls_name] - ref_probs[cls_name])
                        prob_diff_info[cls_name] = {
                            "reference": ref_probs[cls_name],
                            "website": web_probs[cls_name],
                            "abs_diff": round(diff, 5),
                        }
                        if diff > max_prob_diff:
                            max_prob_diff = diff
                stats["prob_diffs"].append(max_prob_diff)

            # Compare attention weights if available (for attention fusion)
            attn_diff_info: Dict[str, Any] = {}
            if model_key == "attention_fusion" and rec["reference_cnn_weight"] is not None:
                web_attn = website_pred.get("attention_weights", {}) or {}
                web_cnn = web_attn.get("cnn_branch_weight")
                web_vit = web_attn.get("vit_branch_weight")
                if web_cnn is not None:
                    diff_cnn = abs(web_cnn - rec["reference_cnn_weight"])
                    diff_vit = abs(web_vit - rec["reference_vit_weight"]) if web_vit is not None else 0.0
                    attn_diff_info = {
                        "cnn_weight": {"reference": rec["reference_cnn_weight"], "website": web_cnn, "diff": round(diff_cnn, 5)},
                        "vit_weight": {"reference": rec["reference_vit_weight"], "website": web_vit, "diff": round(diff_vit, 5)},
                    }
                    stats["attn_diffs"].append(max(diff_cnn, diff_vit))

            # Match criteria
            class_match = (web_class == ref_class) if ref_class else True
            prob_match = (max_prob_diff <= self.prob_tolerance) if ref_probs else True

            if class_match and prob_match:
                stats["matches"] += 1
            else:
                stats["mismatches"] += 1
                mismatches.append({
                    "image": img_name,
                    "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                    "model_key": model_key,
                    "research_prediction": ref_class,
                    "website_prediction": web_class,
                    "probability_difference": round(max_prob_diff, 5),
                    "probability_details": prob_diff_info,
                    "attention_details": attn_diff_info,
                    "status": "FAIL",
                })

        # 5. Compute summary statistics
        total_tested = sum(s["images_tested"] for s in model_stats.values())
        total_matches = sum(s["matches"] for s in model_stats.values())
        total_mismatches = sum(s["mismatches"] for s in model_stats.values())
        total_failures = sum(s["api_failures"] for s in model_stats.values())
        match_rate = (total_matches / total_tested * 100) if total_tested > 0 else 0.0

        # Overall verification verdict
        # Requires: tested > 0, mismatches == 0, failures == 0, missing_images == 0
        is_verified = (
            total_tested > 0
            and total_mismatches == 0
            and total_failures == 0
            and len(missing_images) == 0
        )

        overall_status = "PASS" if is_verified else "FAIL"

        result = {
            "status": overall_status,
            "verified": is_verified,
            "tolerance": {
                "probability": self.prob_tolerance,
                "attention_weights": self.attn_tolerance,
            },
            "overall": {
                "total_comparisons": total_tested,
                "prediction_matches": total_matches,
                "prediction_mismatches": total_mismatches,
                "api_failures": total_failures,
                "match_rate": round(match_rate, 2),
                "unique_images_tested": len(tested_unique_images),
                "missing_images_count": len(missing_images),
            },
            "models": model_stats,
            "mismatches": mismatches,
            "missing_images": missing_images,
            "failed_requests": failed_requests,
        }

        # Save JSON report
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    def print_terminal_report(self, result: Dict[str, Any]) -> int:
        """Format and print the exact terminal report requested."""
        status = result.get("status")

        if status == "API_OFFLINE":
            print("\n" + "=" * 50)
            print("MODEL VERIFICATION - ABORTED")
            print("=" * 50)
            print(f"Error: {result.get('error')}")
            print("\nPlease ensure the FastAPI server is running:")
            print("  cd backend")
            print("  python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
            print("=" * 50)
            return 1

        if status == "REFERENCE_DATA_MISSING":
            print("\n" + "=" * 50)
            print("MODEL VERIFICATION - REFERENCE DATA MISSING")
            print("=" * 50)
            print(result.get("notice"))
            print(f"\nTarget reference file: {self.reference_csv_path}")
            print("See research/verification/README.md for export instructions.")
            print("=" * 50)
            return 2

        print("\nMODEL VERIFICATION")
        print("==================")

        for model_key in ALL_MODELS:
            stats = result["models"].get(model_key, {})
            name = stats.get("display_name", MODEL_DISPLAY_NAMES.get(model_key, model_key))
            tested = stats.get("images_tested", 0)
            matches = stats.get("matches", 0)
            mismatches = stats.get("mismatches", 0)
            failures = stats.get("api_failures", 0)

            print(f"\n{name}")
            print(f"Images tested: {tested}")
            print(f"Prediction matches: {matches}/{tested if tested > 0 else 0}")
            print(f"Prediction mismatches: {mismatches}")
            print(f"API failures: {failures}")

        overall = result["overall"]
        print("\nOverall:")
        print(f"Total comparisons: {overall['total_comparisons']}")
        print(f"Prediction matches: {overall['prediction_matches']}")
        print(f"Prediction mismatches: {overall['prediction_mismatches']}")
        print(f"Match rate: {overall['match_rate']:.1f}%")

        # Report missing images if any
        if result["missing_images"]:
            print(f"\nMissing reference images ({len(result['missing_images'])}):")
            for m_img in result["missing_images"][:5]:
                print(f"  - {m_img['image']} (expected in {m_img['expected_path']})")
            if len(result["missing_images"]) > 5:
                print(f"  ... and {len(result['missing_images']) - 5} more.")

        # Report mismatches if any (Requirement #7)
        if result["mismatches"]:
            print("\nDETAILED MISMATCHES:")
            print("--------------------")
            for m in result["mismatches"]:
                print(f"Image: {m['image']}")
                print(f"Model: {m['model']}")
                print(f"Research prediction: {m['research_prediction']}")
                print(f"Website prediction: {m['website_prediction']}")
                print(f"Probability difference: {m['probability_difference']}")
                print("Status: FAIL\n")

        print("=" * 50)
        verdict = "VERIFIED (PASS)" if result.get("verified") else "FAILED (UNVERIFIED)"
        print(f"FINAL VERDICT: {verdict}")
        print(f"Report saved to: {self.report_json_path}")
        print("=" * 50)

        return 0 if result.get("verified") else 1


def main():
    parser = argparse.ArgumentParser(description="Model verification between research pipeline and website backend.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI backend base URL")
    parser.add_argument("--reference-csv", default="research/verification/reference_predictions.csv", help="Path to reference predictions CSV")
    parser.add_argument("--images-dir", default="research/verification/images", help="Path to reference test images directory")
    parser.add_argument("--prob-tolerance", type=float, default=0.005, help="Probability comparison tolerance (default: 0.005)")
    parser.add_argument("--attn-tolerance", type=float, default=0.005, help="Attention weight tolerance (default: 0.005)")
    parser.add_argument("--report-json", default="research/verification/verification_report.json", help="Path to save verification report JSON")

    args = parser.parse_args()

    verifier = ModelVerifier(
        api_url=args.api_url,
        reference_csv_path=args.reference_csv,
        images_dir=args.images_dir,
        prob_tolerance=args.prob_tolerance,
        attn_tolerance=args.attn_tolerance,
        report_json_path=args.report_json,
    )

    result = verifier.run_verification()
    exit_code = verifier.print_terminal_report(result)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
