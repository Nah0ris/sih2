"""
Final verdict engine — combine all module outputs into a single
pass/fail result with trust score and breakdown.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------

# ELA score above this → suspicious (0–100 scale; higher = more manipulation)
ELA_THRESHOLD = 40.0


def compute_verdict(
    qr_result: dict,
    ela_score: float,
    copy_move_result: dict,
    layout_result: dict,
) -> dict:
    """Combine all module outputs into a final verdict.

    Parameters
    ----------
    qr_result : dict
        Must contain::

            {
                "signature_valid": bool,
                "fields_cross_check": dict   # output of cross_check_fields()
            }

        If QR was unreadable, pass::

            {"signature_valid": False, "fields_cross_check": None,
             "qr_decode_error": "reason"}

    ela_score : float
        Error-level analysis score, 0–100. Higher = more evidence of
        manipulation.

    copy_move_result : dict
        Must contain::

            {"detected": bool, "details": ...}

    layout_result : dict
        Must contain::

            {"valid": bool, "details": ...}

    Returns
    -------
    dict ::

        {
            "status": "GENUINE" | "SUSPICIOUS" | "TAMPERED",
            "trust_score": float,   # 0-100
            "details": { ... }      # all sub-results for the UI
        }
    """
    # --- Unpack inputs ---
    sig_valid = qr_result.get("signature_valid", False)

    cross_check = qr_result.get("fields_cross_check")
    fields_match = cross_check.get("all_match", False) if cross_check else False

    qr_unreadable = "qr_decode_error" in qr_result

    ela_flagged = ela_score > ELA_THRESHOLD

    copy_move_detected = copy_move_result.get("detected", False)

    layout_valid = layout_result.get("valid", True)

    # --- Decision logic ---

    # Layer 1: Hard failures → TAMPERED
    if not sig_valid or not fields_match:
        status = "TAMPERED"
        # Score: 0 for crypto failure, up to 20 for field mismatch only
        if not sig_valid:
            trust_score = 0.0
        else:
            # Signature OK but fields mismatch — slightly less severe
            trust_score = 15.0

    # Layer 2: Soft signals → SUSPICIOUS
    elif ela_flagged or copy_move_detected or not layout_valid:
        status = "SUSPICIOUS"
        # Start at 60, subtract for each red flag
        trust_score = 60.0
        if ela_flagged:
            # Scale down based on how far above threshold
            overshoot = min((ela_score - ELA_THRESHOLD) / (100 - ELA_THRESHOLD), 1.0)
            trust_score -= 15.0 * overshoot
        if copy_move_detected:
            trust_score -= 15.0
        if not layout_valid:
            trust_score -= 5.0
        trust_score = max(trust_score, 25.0)

    # Layer 3: All clean → GENUINE
    else:
        status = "GENUINE"
        # Start at 95 (not 100 — some inherent uncertainty from image quality)
        trust_score = 95.0
        # Slight bonus for low ELA score
        if ela_score < ELA_THRESHOLD * 0.3:
            trust_score = min(trust_score + 5.0, 100.0)

    # --- Assemble details ---
    details = {
        "signature_valid": sig_valid,
        "qr_readable": not qr_unreadable,
        "field_matches": cross_check if cross_check else {},
        "ela_score": ela_score,
        "ela_threshold": ELA_THRESHOLD,
        "ela_flagged": ela_flagged,
        "copy_move_detected": copy_move_detected,
        "layout_valid": layout_valid,
    }
    if qr_unreadable:
        details["qr_decode_error"] = qr_result.get("qr_decode_error", "unknown")

    return {
        "status": status,
        "trust_score": round(trust_score, 1),
        "details": details,
    }
