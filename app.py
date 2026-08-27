"""
DocCop — Aadhaar Document Authenticity Checker

Streamlit app that wires together all real modules:
  - modules.ocr          (Person 2)
  - modules.ela          (Person 3)
  - modules.forensics    (Person 4)
  - modules.qr_crypto    (Person 5 — you)
  - modules.verdict      (Person 5 — you)

Run:  streamlit run app.py
"""

import logging
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Real module imports ───────────────────────────────────────────────────
from modules.ocr import extract_fields as ocr_extract_fields
from modules.ela import generate_ela_heatmap
from modules.forensics import detect_copy_move, check_layout_consistency, detect_photo_tampering
from modules.qr_crypto import (
    decode_qr,
    parse_secure_qr,
    extract_qr_fields,
    cross_check_fields,
    qr_library_used,
    diagnose_scan_issues,
)
from modules.verdict import compute_verdict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Preprocessing stubs (Person 1 has not landed their module yet)
# These are passthrough — replace with real imports when available:
#   from modules.preprocessing import correct_perspective, normalize_lighting, check_quality
# ──────────────────────────────────────────────────────────────────────────

def check_quality(image: np.ndarray) -> tuple[bool, str]:
    """Basic blur check using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, f"Image too blurry (sharpness: {laplacian_var:.1f}, minimum: 50)"
    return True, ""


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Stub — returns image as-is until Person 1 lands their module."""
    return image


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Basic CLAHE normalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline: run all checks on a single image
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(image_np: np.ndarray) -> dict:
    """Run the full verification pipeline and return all results."""
    results = {}

    # 1. Quality gate
    quality_ok, quality_reason = check_quality(image_np)
    results["quality_ok"] = quality_ok
    results["quality_reason"] = quality_reason
    if not quality_ok:
        return results

    # 2. Preprocessing
    corrected = correct_perspective(image_np)
    corrected = normalize_lighting(corrected)
    results["corrected"] = corrected

    # 3. OCR extraction
    try:
        ocr_fields = ocr_extract_fields(corrected)
    except Exception as e:
        logger.warning("OCR extraction failed: %s", e)
        ocr_fields = {"name": "", "dob": "", "gender": "", "aadhaar_number": "", "address": ""}
    results["ocr_fields"] = ocr_fields

    # 4. QR decode + parse + verify + extract
    raw_qr = decode_qr(image_np)
    results["qr_library"] = qr_library_used()

    if raw_qr is not None:
        try:
            data_payload, signature = parse_secure_qr(raw_qr)
            qr_fields = extract_qr_fields(data_payload)

            # Signature verification — gracefully handle missing cert
            try:
                sig_valid = False  # Default to False
                from modules.qr_crypto import verify_signature
                sig_valid = verify_signature(data_payload, signature)
            except FileNotFoundError:
                logger.warning("UIDAI cert not found — skipping signature verification")
                sig_valid = False
            except Exception as e:
                logger.warning("Signature verification error: %s", e)
                sig_valid = False

            field_cross_check = cross_check_fields(qr_fields, ocr_fields)

            results["qr_fields"] = qr_fields
            results["signature_valid"] = sig_valid
            results["field_cross_check"] = field_cross_check
            results["qr_readable"] = True
        except (ValueError, Exception) as e:
            logger.warning("QR parsing failed: %s", e)
            results["qr_readable"] = False
            results["qr_decode_error"] = str(e)
            results["signature_valid"] = False
            results["qr_fields"] = {}
            results["field_cross_check"] = None
    else:
        results["qr_readable"] = False
        results["qr_decode_error"] = "No QR code found in image"
        results["signature_valid"] = False
        results["qr_fields"] = {}
        results["field_cross_check"] = None
        results["diagnostics"] = diagnose_scan_issues(image_np)

    # 5. ELA
    try:
        heatmap, ela_score = generate_ela_heatmap(image_np)
    except Exception as e:
        logger.warning("ELA failed: %s", e)
        heatmap = np.zeros_like(image_np)
        ela_score = 0.0
    results["heatmap"] = heatmap
    results["ela_score"] = ela_score

    # 6. Copy-move detection
    try:
        copy_move_detected, copy_move_regions = detect_copy_move(corrected)
    except Exception as e:
        logger.warning("Copy-move detection failed: %s", e)
        copy_move_detected = False
        copy_move_regions = []
    results["copy_move_detected"] = copy_move_detected
    results["copy_move_regions"] = copy_move_regions

    # 7. Layout consistency
    try:
        layout_result = check_layout_consistency(corrected)
    except Exception as e:
        logger.warning("Layout check failed: %s", e)
        layout_result = {"layout_match": True, "flagged_regions": []}
    results["layout_result"] = layout_result

    # 8. Photo tampering & splicing analysis
    try:
        photo_tamper_result = detect_photo_tampering(image_np)
    except Exception as e:
        logger.warning("Photo tampering check failed: %s", e)
        photo_tamper_result = {"tampering_detected": False, "confidence_score": 0.0, "anomalies": []}
    results["photo_tamper_result"] = photo_tamper_result

    # 9. Final verdict
    qr_result = {
        "signature_valid": results["signature_valid"],
        "fields_cross_check": results["field_cross_check"],
    }
    if not results["qr_readable"]:
        qr_result["qr_decode_error"] = results.get("qr_decode_error", "unknown")

    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=ela_score,
        copy_move_result={"detected": copy_move_detected},
        layout_result={"valid": layout_result.get("layout_match", True)},
        photo_tamper_result=photo_tamper_result,
    )
    results["verdict"] = verdict

    return results


# ──────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DocCop", page_icon="📄", layout="wide")

VERDICT_STYLE = {
    "GENUINE":    ("#1a7f37", "#e6f4ea", "GENUINE"),
    "SUSPICIOUS": ("#9a6700", "#fff8e6", "SUSPICIOUS"),
    "TAMPERED":   ("#c0392b", "#fdeaea", "TAMPERED"),
}

st.title("DocCop")
st.caption(
    "Upload an Aadhaar card image to run OCR, QR signature "
    "verification, error-level analysis, and copy-move/layout forensics."
)

uploaded_file = st.file_uploader("Upload Aadhaar Card", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)

    with st.spinner("Running verification pipeline..."):
        results = run_pipeline(image_np)

    # ── Quality gate ──────────────────────────────────────────────────
    if not results["quality_ok"]:
        st.image(pil_image, caption="Original Upload", width=420)
        st.error(
            f"Image rejected: {results['quality_reason'] or 'Blurry or no Aadhaar card detected.'}"
        )
        st.info("Please retake the image. The document should be clear, well-lit, and fully visible.")
        st.stop()

    # ── Show uploaded image ───────────────────────────────────────────
    st.image(pil_image, caption="Original Upload", width=420)

    # ── Preprocessing ─────────────────────────────────────────────────
    st.subheader("Preprocessing")
    col1, col2 = st.columns(2)
    corrected = results.get("corrected", image_np)
    with col1:
        st.image(image_np, caption="Original", use_container_width=True, channels="BGR")
    with col2:
        st.image(corrected, caption="Corrected + Normalized", use_container_width=True, channels="BGR")

    # ── OCR Fields ────────────────────────────────────────────────────
    st.subheader("OCR Extracted Fields")
    ocr_fields = results.get("ocr_fields", {})
    ocr_rows = [{"Field": k, "Value": v} for k, v in ocr_fields.items()]
    st.table(ocr_rows)

    # ── QR Signature Check ────────────────────────────────────────────
    st.subheader("QR Signature Check")

    if results.get("qr_readable"):
        if results.get("signature_valid"):
            st.success("Signature valid -- UIDAI public key match")
        else:
            st.warning("QR decoded but signature could not be verified (UIDAI cert may be missing)")

        qr_fields = results.get("qr_fields", {})
        field_cross_check = results.get("field_cross_check", {})

        # Build the field comparison table
        compare_keys = ["name", "dob", "gender", "aadhaar_number", "address"]
        match_rows = []
        for k in compare_keys:
            fc = field_cross_check.get(k, {})
            match_rows.append({
                "Field": k,
                "QR Value": fc.get("qr", qr_fields.get(k, "")),
                "OCR Value": fc.get("ocr", ocr_fields.get(k, "")),
                "Match": "Yes" if fc.get("match", False) else "No",
            })
        st.table(match_rows)

        lib = results.get("qr_library", "unknown")
        st.caption(f"QR decoded using: {lib}")
    else:
        error_msg = results.get("qr_decode_error", "Unknown error")
        st.warning(f"⚠️ QR code could not be detected: {error_msg}")
        
        diag = results.get("diagnostics", {})
        if diag:
            with st.expander("🔍 Image Quality Diagnostics & Retake Tips", expanded=True):
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Sharpness", f"{diag.get('sharpness', 0):.0f}/100", 
                              help="Above 60 is recommended for dense Aadhaar QR codes")
                with col_d2:
                    st.metric("Glare Detected", f"{diag.get('glare_percentage', 0):.1f}%", 
                              help="Under 3% is recommended. Avoid overhead lights on plastic cards.")
                with col_d3:
                    st.metric("Resolution", diag.get("resolution", "N/A"))
                
                st.markdown("**How to get a clean scan:**")
                for tip in diag.get("actionable_tips", []):
                    st.markdown(f"- {tip}")
                st.info("💡 **Pro-Tip:** Aadhaar Secure QR codes contain thousands of microscopic modules. If the phone was too far away, take a closer photo or crop directly around the QR box.")

    # ── ELA ────────────────────────────────────────────────────────────
    st.subheader("Visual Forensics -- Error Level Analysis")
    c1, c2 = st.columns([2, 1])
    with c1:
        heatmap = results.get("heatmap", np.zeros_like(image_np))
        st.image(heatmap, caption="ELA Heatmap", use_container_width=True, channels="BGR")
    with c2:
        ela_score = results.get("ela_score", 0.0)
        st.metric("ELA Error Score", f"{ela_score:.1f}", help="Higher = more suspicious")

    # ── Copy-move + layout ────────────────────────────────────────────
    st.subheader("Layout & Duplication Check")
    lc1, lc2 = st.columns(2)
    with lc1:
        if results.get("copy_move_detected"):
            regions = results.get("copy_move_regions", [])
            st.error(f"Copy-move duplication detected in {len(regions)} region(s)")
        else:
            st.success("No copy-move duplication detected")
    with lc2:
        layout_result = results.get("layout_result", {})
        if layout_result.get("layout_match", True):
            st.success("Layout matches expected template")
        else:
    # ── Photo Box Splicing & Face Forensics ─────────────────────────
    st.subheader("Photo & Face Splicing Forensics")
    photo_res = results.get("photo_tamper_result", {})
    if photo_res.get("tampering_detected"):
        st.error(f"❌ Digital Photo Tampering Detected (Anomaly Score: {photo_res.get('confidence_score', 0):.0f}/100)")
        for a in photo_res.get("anomalies", []):
            st.markdown(f"- ⚠️ {a}")
    else:
        st.success("✅ No photo splicing or sensor noise anomalies detected in ID photo box")

    # ── Final Verdict ─────────────────────────────────────────────────
    st.subheader("Final Verdict")
    verdict = results.get("verdict", {"status": "SUSPICIOUS", "trust_score": 0.0, "details": {}})
    color, bg, label = VERDICT_STYLE.get(verdict["status"], ("#333", "#eee", "UNKNOWN"))
    trust = verdict.get("trust_score", 0.0)
    tier = verdict.get("confidence_tier", "UNKNOWN")
    summary = verdict.get("summary", "")

    tier_badge = "🔐 TIER 1: CRYPTOGRAPHIC PROOF" if tier == "CRYPTOGRAPHIC" else "🔍 TIER 2/3: VISUAL FORENSICS ONLY"

    st.markdown(
        f"""
        <div style="background-color:{bg}; border:2px solid {color}; border-radius:10px;
                    padding:20px; text-align:center;">
            <span style="background-color:{color}; color:white; padding:4px 12px; border-radius:15px; font-size:12px; font-weight:bold; letter-spacing:1px;">
                {tier_badge}
            </span>
            <h2 style="color:{color}; margin:10px 0 5px 0;">{label}</h2>
            <p style="font-size:18px; margin:4px 0 8px 0;">Trust Score: <b>{trust:.0f}/100</b></p>
            <p style="font-size:14px; color:#555; margin:0 auto; max-width:600px;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Full details (debug)"):
        st.json(verdict.get("details", {}))

else:
    st.info("Upload a JPG or PNG of an Aadhaar card to begin.")
