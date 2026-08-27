"""
Final verdict engine — combine all module outputs into a single
pass/fail result with trust score and breakdown using a tiered confidence model.
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
    photo_tamper_result: dict = None,
) -> dict:
    """Combine all module outputs into a final verdict using a Tiered Confidence Model.

    Tiers:
      - Tier 1 (CRYPTOGRAPHIC): QR is readable, signature verified with UIDAI,
        fields cross-checked. High confidence (Trust score: 0-100).
      - Tier 2 / 3 (FORENSIC_ONLY): QR is unreadable or absent (damaged, glare/blurry photo).
        Visual forensics only. Never overclaims 'GENUINE'; detects photo tampering / copy-move / ELA,
        and outputs 'TAMPERED' on detected visual splices, or 'SUSPICIOUS' (Unverified Clean) if clean.
    """
    # --- Unpack inputs ---
    sig_valid = qr_result.get("signature_valid", False)
    cross_check = qr_result.get("fields_cross_check")
    fields_match = cross_check.get("all_match", False) if cross_check else False
    qr_unreadable = "qr_decode_error" in qr_result

    ela_flagged = ela_score > ELA_THRESHOLD
    copy_move_detected = copy_move_result.get("detected", False)
    layout_valid = layout_result.get("valid", True)
    
    photo_tamper = photo_tamper_result or {}
    photo_tampered = photo_tamper.get("tampering_detected", False)

    # --- Tier 1: Cryptographic Verification Available ---
    if not qr_unreadable:
        confidence_tier = "CRYPTOGRAPHIC"

        # Cryptographic signature invalid -> definite forgery
        if not sig_valid:
            status = "TAMPERED"
            trust_score = 0.0
            summary = "Cryptographic digital signature is INVALID. Document QR is forged or corrupted."

        # QR Signature valid, but card printed fields mismatch QR
        elif not fields_match:
            status = "TAMPERED"
            trust_score = 15.0
            summary = "Field mismatch detected: Printed card text does not match cryptographically signed QR data."

        # Photo splice detected
        elif photo_tampered:
            status = "TAMPERED"
            trust_score = 10.0
            reasons = ", ".join(photo_tamper.get("anomalies", []))
            summary = f"Photo tampering detected: {reasons or 'Splicing artifacts in ID photo box.'}"

        # Signature valid + fields match + soft signals
        elif ela_flagged or copy_move_detected or not layout_valid:
            status = "SUSPICIOUS"
            trust_score = 60.0
            if ela_flagged:
                overshoot = min((ela_score - ELA_THRESHOLD) / (100 - ELA_THRESHOLD), 1.0)
                trust_score -= 15.0 * overshoot
            if copy_move_detected:
                trust_score -= 15.0
            if not layout_valid:
                trust_score -= 5.0
            trust_score = max(trust_score, 25.0)
            summary = "Cryptographic signature is valid, but visual forensics detected anomalies (compression/duplication/layout)."

        else:
            status = "GENUINE"
            trust_score = 95.0
            if ela_score < ELA_THRESHOLD * 0.3:
                trust_score = min(trust_score + 5.0, 100.0)
            summary = "All cryptographic and visual forensic checks passed cleanly. Document is verified genuine."

    # --- Tier 2 / 3: Forensic-Only (QR Unreadable / Photo Glare / Damaged) ---
    else:
        confidence_tier = "FORENSIC_ONLY"

        if photo_tampered:
            status = "TAMPERED"
            trust_score = 12.0
            reasons = ", ".join(photo_tamper.get("anomalies", []))
            summary = f"Photo tampering detected: {reasons or 'Digital face swap / splicing artifacts identified in photo box.'}"

        elif copy_move_detected or ela_flagged:
            status = "SUSPICIOUS"
            trust_score = 35.0
            summary = "QR unreadable + visual tampering detected (ELA or copy-move duplication flagged)."

        elif not layout_valid:
            status = "SUSPICIOUS"
            trust_score = 45.0
            summary = "QR unreadable + card layout does not match standard Aadhaar template."

        else:
            status = "SUSPICIOUS"
            trust_score = 65.0
            summary = "No visual tampering detected, but cryptographic QR check was unavailable (photo quality/glare). Verification unconfirmed."

    # --- Assemble details ---
    details = {
        "signature_valid": sig_valid,
        "qr_readable": not qr_unreadable,
        "confidence_tier": confidence_tier,
        "field_matches": cross_check if cross_check else {},
        "ela_score": ela_score,
        "ela_threshold": ELA_THRESHOLD,
        "ela_flagged": ela_flagged,
        "copy_move_detected": copy_move_detected,
        "layout_valid": layout_valid,
        "photo_tampering": photo_tamper,
    }
    if qr_unreadable:
        details["qr_decode_error"] = qr_result.get("qr_decode_error", "unknown")

    return {
        "status": status,
        "confidence_tier": confidence_tier,
        "trust_score": round(trust_score, 1),
        "summary": summary,
        "details": details,
    }
