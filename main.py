"""
main.py — Entry point for the Aadhaar Document Verification System.

Usage:
  1. Streamlit UI (recommended):
       streamlit run app.py

  2. CLI single-image check:
       python main.py test_data/genuine_sample.jpg

  3. As a Python module:
       from main import verify_document
       result = verify_document("path/to/image.jpg")
"""

import json
import logging
import sys

import cv2
import numpy as np

from modules.ocr import extract_fields as ocr_extract_fields
from modules.ela import generate_ela_heatmap
from modules.forensics import detect_copy_move, check_layout_consistency
from modules.qr_crypto import (
    decode_qr,
    parse_secure_qr,
    extract_qr_fields,
    cross_check_fields,
    qr_library_used,
)
from modules.verdict import compute_verdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Preprocessing stubs (Person 1 has not landed their module yet)
# ──────────────────────────────────────────────────────────────────────────

def check_quality(image: np.ndarray) -> tuple[bool, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, f"Image too blurry (sharpness: {laplacian_var:.1f}, minimum: 50)"
    return True, ""


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────────────────────────────────
# Full verification pipeline
# ──────────────────────────────────────────────────────────────────────────

def verify_document(image_path: str) -> dict:
    """Run the complete verification pipeline on a single image.

    Returns a dict with the final verdict and all intermediate results.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {
            "error": f"Could not load image: {image_path}",
            "verdict": {"status": "TAMPERED", "trust_score": 0.0, "details": {}},
        }

    # 1. Quality gate
    quality_ok, quality_reason = check_quality(image)
    if not quality_ok:
        return {
            "error": f"Quality check failed: {quality_reason}",
            "verdict": {"status": "TAMPERED", "trust_score": 0.0, "details": {}},
        }

    # 2. Preprocessing
    corrected = normalize_lighting(image)

    # 3. OCR
    try:
        ocr_fields = ocr_extract_fields(corrected)
        logger.info("OCR fields: %s", ocr_fields)
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        ocr_fields = {"name": "", "dob": "", "gender": "", "aadhaar_number": "", "address": ""}

    # 4. QR decode + parse + verify
    raw_qr = decode_qr(image)
    logger.info("QR library used: %s", qr_library_used())

    qr_fields = {}
    sig_valid = False
    field_cross_check = None
    qr_readable = False
    qr_error = None

    if raw_qr is not None:
        try:
            data_payload, signature = parse_secure_qr(raw_qr)
            qr_fields = extract_qr_fields(data_payload)
            logger.info("QR fields: %s", qr_fields)

            try:
                from modules.qr_crypto import verify_signature
                sig_valid = verify_signature(data_payload, signature)
            except FileNotFoundError:
                logger.warning("UIDAI cert not found -- signature check skipped")
                sig_valid = False
            except Exception as e:
                logger.warning("Signature verification error: %s", e)
                sig_valid = False

            field_cross_check = cross_check_fields(qr_fields, ocr_fields)
            qr_readable = True
            logger.info("Field cross-check: %s", field_cross_check)
        except (ValueError, Exception) as e:
            logger.warning("QR parsing failed: %s", e)
            qr_error = str(e)
    else:
        qr_error = "No QR code found in image"
        logger.warning(qr_error)

    # 5. ELA
    try:
        _, ela_score = generate_ela_heatmap(image)
        logger.info("ELA score: %.2f", ela_score)
    except Exception as e:
        logger.warning("ELA failed: %s", e)
        ela_score = 0.0

    # 6. Copy-move
    try:
        copy_move_detected, copy_move_regions = detect_copy_move(corrected)
        logger.info("Copy-move detected: %s (%d regions)", copy_move_detected, len(copy_move_regions))
    except Exception as e:
        logger.warning("Copy-move failed: %s", e)
        copy_move_detected = False
        copy_move_regions = []

    # 7. Layout
    try:
        layout_result = check_layout_consistency(corrected)
        logger.info("Layout: %s", layout_result)
    except Exception as e:
        logger.warning("Layout check failed: %s", e)
        layout_result = {"layout_match": True, "flagged_regions": []}

    # 8. Verdict
    qr_result = {
        "signature_valid": sig_valid,
        "fields_cross_check": field_cross_check,
    }
    if not qr_readable:
        qr_result["qr_decode_error"] = qr_error or "unknown"

    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=ela_score,
        copy_move_result={"detected": copy_move_detected},
        layout_result={"valid": layout_result.get("layout_match", True)},
    )

    return {
        "image_path": image_path,
        "quality_ok": quality_ok,
        "ocr_fields": ocr_fields,
        "qr_readable": qr_readable,
        "qr_library": qr_library_used(),
        "qr_fields": qr_fields,
        "signature_valid": sig_valid,
        "field_cross_check": field_cross_check,
        "ela_score": ela_score,
        "copy_move_detected": copy_move_detected,
        "layout_match": layout_result.get("layout_match", True),
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _make_serializable(obj):
    """Convert numpy types and other non-serializable objects for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return "<image data>"
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        print("   or: streamlit run app.py")
        sys.exit(1)

    image_path = sys.argv[1]
    result = verify_document(image_path)

    # Pretty-print the result as JSON
    serializable = _make_serializable(result)
    print("\n" + "=" * 60)
    print("  VERIFICATION RESULT")
    print("=" * 60)
    print(json.dumps(serializable, indent=2, ensure_ascii=False))

    verdict = result["verdict"]
    status = verdict["status"]
    score = verdict["trust_score"]
    print(f"\n  >> {status} (Trust Score: {score:.0f}/100)")
    print("=" * 60)

    sys.exit(0 if status == "GENUINE" else 1)
