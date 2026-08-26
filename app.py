"""
Document Authenticity Checker — Frontend (Person 5)

Built entirely against dummy data that matches the exact output shapes
every other module contract promises. Swap the DUMMY_* functions below
for real imports once modules/*.py land on main — nothing else in this
file should need to change if the contracts are followed.

Real imports would look like:
    from modules.preprocessing import correct_perspective, normalize_lighting, check_quality
    from modules.ocr import extract_fields
    from modules.ela import generate_ela_heatmap
    from modules.forensics import detect_copy_move, check_layout_consistency
    from modules.qr_crypto import decode_qr, verify_signature, extract_qr_fields, cross_check_fields
    from modules.verdict import compute_verdict
"""

import time
import numpy as np
import streamlit as st
from PIL import Image

# Exact UI contract data used for the prototype display.
QUALITY_RESULT = (True, "")
QR_FIELDS = {
    "name": "RAHUL SHARMA",
    "dob": "15/08/2000",
    "gender": "Male",
    "aadhaar_number": "1234 5678 9012",
    "address": "123 MG Road, Bangalore",
}
OCR_FIELDS = {
    "name": "RAHUL SHARMA",
    "dob": "15/08/2000",
    "gender": "Male",
    "aadhaar_number": "1234 5678 9012",
    "address": "123 MG Road, Bangalore",
}
SIGNATURE_VALID = True
ELA_SCORE = 12.4
COPY_MOVE_DETECTED = False
LAYOUT_MATCH = True
VERDICT = {
    "status": "GENUINE",
    "trust_score": 94.5,
    "details": {},
}

# ──────────────────────────────────────────────────────────────────────────
# DUMMY BACKEND — replace each function body with the real module call.
# Keep function names/signatures identical so integration is a 1-line swap.
# ──────────────────────────────────────────────────────────────────────────

def dummy_check_quality(image: np.ndarray) -> tuple[bool, str]:
    return QUALITY_RESULT

def dummy_correct_perspective(image: np.ndarray) -> np.ndarray:
    return image

def dummy_normalize_lighting(image: np.ndarray) -> np.ndarray:
    return image

def dummy_extract_fields(image: np.ndarray) -> dict:
    return OCR_FIELDS.copy()

def dummy_decode_qr(image: np.ndarray) -> bytes | None:
    return b"dummy_qr_payload"

def dummy_verify_signature(qr_data: bytes) -> bool:
    return SIGNATURE_VALID

def dummy_extract_qr_fields(qr_data: bytes) -> dict:
    return QR_FIELDS.copy()

def dummy_cross_check_fields(qr_fields: dict, ocr_fields: dict) -> dict:
    return {k: (qr_fields.get(k) == ocr_fields.get(k)) for k in ocr_fields}

def dummy_generate_ela_heatmap(image: np.ndarray) -> tuple[np.ndarray, float]:
    # fake a colorful heatmap-ish array just so st.image has something to show
    h, w = image.shape[:2] if image.ndim >= 2 else (300, 480)
    heat = np.random.randint(0, 60, (h, w, 3), dtype=np.uint8)
    heat[:, :, 0] = 255 - heat[:, :, 0]  # bias toward a JET-like look
    return heat, ELA_SCORE

def dummy_detect_copy_move(image: np.ndarray) -> tuple[bool, list]:
    return (COPY_MOVE_DETECTED, [])

def dummy_check_layout_consistency(image: np.ndarray) -> dict:
    return {"layout_match": LAYOUT_MATCH, "flagged_regions": []}

def dummy_compute_verdict(qr_result, ela_score, copy_move_result, layout_result) -> dict:
    sig_ok = qr_result["signature_valid"]
    fields_ok = all(qr_result["field_matches"].values())
    copy_move_flagged = copy_move_result[0]
    layout_ok = layout_result["layout_match"]

    if not sig_ok or not fields_ok:
        status, score = "TAMPERED", 12.0
    elif ela_score > 25 or copy_move_flagged or not layout_ok:
        status, score = "SUSPICIOUS", 58.0
    else:
        status, score = "GENUINE", 96.0

    return {
        "status": status,
        "trust_score": score,
        "details": {
            "qr_signature_valid": sig_ok,
            "field_matches": qr_result["field_matches"],
            "ela_score": ela_score,
            "copy_move_detected": copy_move_flagged,
            "layout_match": layout_ok,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + STYLE
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DocCop", page_icon="📄", layout="wide")

VERDICT_STYLE = {
    "GENUINE": ("#1a7f37", "#e6f4ea", "✅ GENUINE"),
    "SUSPICIOUS": ("#9a6700", "#fff8e6", "⚠️ SUSPICIOUS"),
    "TAMPERED": ("#c0392b", "#fdeaea", "❌ TAMPERED"),
}

st.title("DocCop")
st.caption("Upload an Aadhaar card image to run perspective correction, OCR, QR signature "
           "verification, error-level analysis, and copy-move/layout forensics.")

uploaded_file = st.file_uploader("Upload Aadhaar Card", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)
    quality_ok, quality_reason = dummy_check_quality(image_np)
    corrected = dummy_correct_perspective(image_np)
    corrected = dummy_normalize_lighting(corrected)
    ocr_fields = dummy_extract_fields(corrected)
    qr_data = dummy_decode_qr(image_np)
    if qr_data is None:
        signature_valid = False
        qr_fields = {}
        field_matches = {k: False for k in ocr_fields}
    else:
        signature_valid = dummy_verify_signature(qr_data)
        qr_fields = dummy_extract_qr_fields(qr_data)
        field_matches = dummy_cross_check_fields(qr_fields, ocr_fields)
    qr_result = {"signature_valid": signature_valid, "field_matches": field_matches, "qr_fields": qr_fields}
    heatmap, ela_score = dummy_generate_ela_heatmap(image_np)
    copy_move_result = dummy_detect_copy_move(corrected)
    layout_result = dummy_check_layout_consistency(corrected)
    verdict = dummy_compute_verdict(qr_result, ela_score, copy_move_result, layout_result)

    st.image(pil_image, caption="Original Upload", width=420)

    # ── Quality gate ──────────────────────────────────────────────────
    with st.spinner("Checking image quality..."):
        time.sleep(0.2)

    if not quality_ok:
        st.error(f"Image rejected: {quality_reason or 'Blurry or no Aadhaar card detected.'}")
        st.info("Please retake the image. The document should be clear, well-lit, and fully visible.")
        st.stop()

    # ── Preprocessing ─────────────────────────────────────────────────
    st.subheader("Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original", use_container_width=True)
    with col2:
        st.image(corrected, caption="Corrected + Normalized", use_container_width=True)

    # ── QR Signature Check ──────────────────────────────────────────────
    st.subheader("QR Signature Check")
    if signature_valid:
        st.success("✅ Signature valid — UIDAI public key match")
    else:
        st.error("❌ Signature invalid or QR not found")

    match_rows = [
        {"Field": k, "QR Value": qr_fields.get(k, ""), "OCR Value": ocr_fields.get(k, ""),
         "Match": "✅" if v else "❌"}
        for k, v in field_matches.items()
    ]
    st.table(match_rows)

    # ── ELA ────────────────────────────────────────────────────────────
    st.subheader("Visual Forensics — Error Level Analysis")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(heatmap, caption="ELA Heatmap", use_container_width=True)
    with c2:
        st.metric("ELA Error Score", f"{ela_score:.1f}", help="Higher = more suspicious")

    # ── Copy-move + layout ────────────────────────────────────────────
    st.subheader("Layout & Duplication Check")
    lc1, lc2 = st.columns(2)
    with lc1:
        if copy_move_result[0]:
            st.error(f"⚠️ Copy-move duplication detected in {len(copy_move_result[1])} region(s)")
            st.write(copy_move_result[1])
        else:
            st.success("✅ No copy-move duplication detected")
    with lc2:
        if layout_result["layout_match"]:
            st.success("✅ Layout matches expected template")
        else:
            st.error("⚠️ Layout mismatch detected")
            st.write(layout_result["flagged_regions"])

    st.subheader("Final Verdict")
    color, bg, label = VERDICT_STYLE[verdict["status"]]
    st.markdown(
        f"""
        <div style="background-color:{bg}; border:2px solid {color}; border-radius:10px;
                    padding:20px; text-align:center;">
            <h2 style="color:{color}; margin:0;">{label}</h2>
            <p style="font-size:18px; margin:8px 0 0 0;">Trust Score: <b>{verdict['trust_score']:.0f}/100</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Full details (debug)"):
        st.json(verdict["details"])

else:
    st.info("Upload a JPG or PNG of an Aadhaar card to begin.")
