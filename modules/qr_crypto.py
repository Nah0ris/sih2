"""
Aadhaar Secure QR Code — decode, parse, verify, cross-check.

Implements the UIDAI Secure QR Code binary format:
  Base-10 numeric string → BigInteger → GZIP decompress →
  0xFF-delimited text fields + JPEG2000 photo + SHA-256 hashes +
  256-byte RSA digital signature.

Reference: pyaadhaar (github.com/tanmoysrt/pyaadhaar) confirmed against
multiple independent sources.  The official UIDAI PDF spec
(User_manulal_QR_Code_15032019.pdf) is no longer publicly hosted.
"""

from __future__ import annotations

import logging
import re
import zlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered field names in the decompressed payload, separated by 0xFF.
# Newer cards may prepend a "Vx" version marker — handled in _detect_version.
_BASE_FIELDS = [
    "email_mobile_status",
    "referenceid",
    "name",
    "dob",
    "gender",
    "careof",
    "district",
    "landmark",
    "house",
    "location",
    "pincode",
    "postoffice",
    "state",
    "street",
    "subdistrict",
    "vtc",
]

_SIGNATURE_LEN = 256        # RSA-2048 → 256-byte signature
_HASH_LEN = 32              # SHA-256 → 32 bytes per hash

# Path to UIDAI public certificate — user must supply this file.
_CERT_PATH = Path(__file__).resolve().parent.parent / "assets" / "uidai_public_cert.pem"

# Gender code mapping (QR stores single-letter codes)
_GENDER_MAP = {"M": "Male", "F": "Female", "T": "Other"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. decode_qr  — scan the QR code from an image
# ──────────────────────────────────────────────────────────────────────────────

# Track which library is actually used, exposed for caller inspection.
_qr_library_used: Optional[str] = None


def _decode_with_pyzbar(image: np.ndarray) -> Optional[str]:
    """Attempt QR decode using pyzbar (wraps the zbar C library)."""
    global _qr_library_used
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    except ImportError as exc:
        logger.warning("pyzbar import failed: %s", exc)
        return None

    try:
        results = pyzbar_decode(image, symbols=[ZBarSymbol.QRCODE])
    except Exception as exc:
        logger.debug("pyzbar decode raised: %s", exc)
        return None

    if not results:
        return None

    _qr_library_used = "pyzbar"
    return results[0].data.decode("utf-8", errors="replace")


def _decode_with_opencv(image: np.ndarray) -> Optional[str]:
    """Fallback: decode QR using OpenCV's built-in QRCodeDetector."""
    global _qr_library_used
    try:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(image)
    except Exception as exc:
        logger.debug("cv2.QRCodeDetector raised: %s", exc)
        return None

    if not data:
        return None

    _qr_library_used = "cv2.QRCodeDetector"
    return data


def _try_single_image(img: np.ndarray) -> Optional[str]:
    """Try all decoders on a single image variant."""
    res = _decode_with_pyzbar(img)
    if res is not None:
        return res
    return _decode_with_opencv(img)


def _generate_qr_crops(gray: np.ndarray) -> list[np.ndarray]:
    """Generate intelligent candidate crops where Aadhaar QR is typically found."""
    h, w = gray.shape[:2]
    crops = []

    # Aadhaar QR is usually in bottom-right, bottom-left, or right-half
    # Quadrant crops (with generous margins)
    crops.append(gray[int(h * 0.3):, int(w * 0.4):])       # Bottom-right
    crops.append(gray[int(h * 0.3):, :int(w * 0.6)])       # Bottom-left
    crops.append(gray[:int(h * 0.7), int(w * 0.4):])       # Top-right
    crops.append(gray[int(h * 0.2):int(h * 0.9), int(w * 0.3):int(w * 0.95)]) # Center-right

    # Contour-based square detection for high-density QR boxes
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.02) and area < (h * w * 0.6):  # reasonable QR size
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = float(cw) / ch if ch > 0 else 0
                if 0.75 <= aspect <= 1.35:  # roughly square
                    pad = int(min(cw, ch) * 0.15)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + cw + pad)
                    y2 = min(h, y + ch + pad)
                    crops.append(gray[y1:y2, x1:x2])
    except Exception:
        pass

    return [c for c in crops if c.size > 0 and c.shape[0] > 60 and c.shape[1] > 60]


def decode_qr(image: np.ndarray) -> Optional[str]:
    """Decode the Aadhaar Secure QR Code from an image using multi-stage recovery.

    Aadhaar Secure QR codes are dense (version 15-20+) and standard phone scans
    frequently fail without contrast enhancement, rescaling, or region extraction.

    Recovery stages:
      1. Direct raw and grayscale scan
      2. Multi-scale variations (0.5x, 0.75x, 1.5x, 2.0x Lanczos)
      3. Image enhancements (CLAHE contrast, Sharpening, Adaptive Thresholding, Otsu)
      4. Targeted region crops (Quadrants and detected square contours)
      5. 90/180/270 degree rotations

    Returns the raw decoded payload string, or None if all stages fail.
    """
    if image is None or image.size == 0:
        logger.error("decode_qr received a None or empty image")
        return None

    # --- Stage 1: Direct scan ---
    res = _try_single_image(image)
    if res is not None:
        return res

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    res = _try_single_image(gray)
    if res is not None:
        return res

    # --- Stage 2: CLAHE Contrast Enhancement & Sharpening ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    res = _try_single_image(enhanced_gray)
    if res is not None:
        return res

    # Sharpening kernel
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_gray, -1, kernel_sharpen)
    res = _try_single_image(sharpened)
    if res is not None:
        return res

    # Adaptive Thresholding (Otsu & Gaussian)
    _, otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = _try_single_image(otsu)
    if res is not None:
        return res

    adapt_thresh = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    res = _try_single_image(adapt_thresh)
    if res is not None:
        return res

    # --- Stage 3: Rescaling Variants (Aadhaar QR needs 2x Lanczos for dense modules) ---
    h, w = gray.shape[:2]
    scales = [1.5, 2.0, 0.75, 0.5]
    for s in scales:
        resized = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
        res = _try_single_image(resized)
        if res is not None:
            return res
        # Also try sharpened on 2x
        if s == 2.0:
            res = _try_single_image(clahe.apply(resized))
            if res is not None:
                return res

    # --- Stage 4: Targeted Region Crops (Sub-regions & Contours) ---
    candidate_crops = _generate_qr_crops(gray)
    for crop in candidate_crops:
        # Try raw crop
        res = _try_single_image(crop)
        if res is not None:
            return res
        # Try 2x upscale on the crop
        ch, cw = crop.shape[:2]
        crop_2x = cv2.resize(crop, (cw * 2, ch * 2), interpolation=cv2.INTER_LANCZOS4)
        res = _try_single_image(crop_2x)
        if res is not None:
            return res
        # Try enhanced crop
        crop_enh = clahe.apply(crop_2x)
        res = _try_single_image(crop_enh)
        if res is not None:
            return res
        _, crop_otsu = cv2.threshold(crop_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = _try_single_image(crop_otsu)
        if res is not None:
            return res

    # --- Stage 5: Rotations (for portrait / landscape photo orientations) ---
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rot_img = cv2.rotate(gray, rot)
        res = _try_single_image(rot_img)
        if res is not None:
            return res
        res = _try_single_image(clahe.apply(rot_img))
        if res is not None:
            return res

    logger.warning("No QR code found in image after all 5 recovery stages.")
    return None


def qr_library_used() -> Optional[str]:
    """Return which library last successfully decoded a QR code."""
    return _qr_library_used


# ──────────────────────────────────────────────────────────────────────────────
# 2. parse_secure_qr  — Base-10 string → decompressed bytes + signature
# ──────────────────────────────────────────────────────────────────────────────

def parse_secure_qr(raw_data: str) -> tuple[bytes, bytes]:
    """Convert raw QR string to (data_payload, signature_bytes).

    Steps:
      1. Convert Base-10 numeric string → Python int → big-endian bytes
      2. GZIP-decompress
      3. Split: everything except last 256 bytes = data,
                last 256 bytes = RSA signature

    Raises ValueError if the data is not a valid Secure QR payload.
    """
    # --- Step 1: Base-10 → bytes ---
    raw_data = raw_data.strip()
    if not raw_data.isdigit():
        raise ValueError(
            "QR payload is not a Base-10 numeric string — "
            "this may be an old XML-format QR, which is out of scope."
        )

    big_int = int(raw_data)
    # Compute the exact byte length needed (no over-allocation)
    byte_length = (big_int.bit_length() + 7) // 8
    raw_bytes = big_int.to_bytes(byte_length, byteorder="big")

    # --- Step 2: GZIP decompress ---
    try:
        decompressed = zlib.decompress(raw_bytes, 16 + zlib.MAX_WBITS)
    except zlib.error as exc:
        raise ValueError(f"GZIP decompression failed: {exc}") from exc

    # --- Step 3: Split signature ---
    if len(decompressed) < _SIGNATURE_LEN + 10:
        raise ValueError(
            f"Decompressed payload too short ({len(decompressed)} bytes) "
            f"to contain a {_SIGNATURE_LEN}-byte signature + data."
        )

    data_payload = decompressed[:-_SIGNATURE_LEN]
    signature = decompressed[-_SIGNATURE_LEN:]

    return data_payload, signature


# ──────────────────────────────────────────────────────────────────────────────
# 3. extract_qr_fields  — parse decompressed payload into a dict
# ──────────────────────────────────────────────────────────────────────────────

def _detect_version(data: bytes) -> tuple[int, int]:
    """Detect optional version marker at the start of the payload.

    Returns (version_number, offset) where offset is the number of bytes
    to skip before reading the first 0xFF-delimited field.
    Version 0 means no version marker was found.
    """
    # Version markers look like b'V2\xff' or b'V3\xff'
    if len(data) >= 3 and data[0:1] == b"V" and data[2:3] == b"\xff":
        try:
            version = int(chr(data[1]))
            return version, 2  # skip "Vx", the \xff is a normal delimiter
        except (ValueError, IndexError):
            pass
    return 0, 0


def _normalize_dob(raw_dob: str) -> str:
    """Normalize DOB to DD/MM/YYYY regardless of input format.

    Handles: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, YYYY/MM/DD.
    Returns the original string if it cannot be parsed.
    """
    raw_dob = raw_dob.strip()

    # DD-MM-YYYY or DD/MM/YYYY
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$", raw_dob)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$", raw_dob)
    if m:
        return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"

    logger.warning("Could not normalize DOB format: %r", raw_dob)
    return raw_dob


def _normalize_gender(raw_gender: str) -> str:
    """Normalize gender to 'Male' / 'Female' / 'Other'."""
    g = raw_gender.strip().upper()
    if g in _GENDER_MAP:
        return _GENDER_MAP[g]
    # Already full word?
    if g in ("MALE", "FEMALE", "OTHER"):
        return g.capitalize()
    logger.warning("Unknown gender code: %r, returning as-is", raw_gender)
    return raw_gender.strip()


def extract_qr_fields(data_payload: bytes) -> dict:
    """Parse the decompressed Secure QR payload into demographic fields.

    Parameters
    ----------
    data_payload : bytes
        The decompressed payload **without** the trailing 256-byte signature.
        This is the first element returned by ``parse_secure_qr()``.

    Returns
    -------
    dict with keys: name, dob, gender, aadhaar_number, address
    """
    # Detect version marker
    version, offset = _detect_version(data_payload)
    if version:
        logger.info("Detected Secure QR version: V%d", version)

    # Determine how many trailing bytes to strip before splitting fields.
    # After the text fields come: photo + email_hash + mobile_hash.
    # We need to figure out where text fields end.
    # Strategy: find ALL 0xFF positions, take the first len(fields) as
    # delimiters between text fields. Everything after the last text
    # delimiter is photo + hashes (which we don't need for demographic
    # extraction).

    fields = _BASE_FIELDS[:]
    if version:
        fields = ["version"] + fields + ["last_4_digits_mobile_no"]

    # Find all 0xFF byte positions
    delimiters = []
    for i in range(offset, len(data_payload)):
        if data_payload[i] == 0xFF:
            delimiters.append(i)

    # We need at least len(fields) - 1 delimiters to separate len(fields)
    # fields.  The first field starts at `offset` and runs to the first
    # delimiter.  Actually: we need len(fields) delimiters because the
    # pattern is: field0 | 0xFF | field1 | 0xFF | ... | 0xFF | fieldN | 0xFF
    # followed by photo bytes.
    # Wait — looking at pyaadhaar more carefully: it prepends -1 to the
    # delimiter list and uses delimeter[i]+1 : delimeter[i+1].
    # So the first field is data[0:delim[0]], second is data[delim[0]+1:delim[1]], etc.
    # We need len(fields) delimiters to extract len(fields) fields.

    if len(delimiters) < len(fields):
        raise ValueError(
            f"Expected at least {len(fields)} 0xFF delimiters but found "
            f"{len(delimiters)}. Payload may be malformed or a different format."
        )

    # Extract text fields
    parsed = {}
    # First field: from offset to first delimiter
    parsed[fields[0]] = data_payload[offset:delimiters[0]].decode(
        "ISO-8859-1", errors="replace"
    )
    # Remaining fields
    for i in range(1, len(fields)):
        start = delimiters[i - 1] + 1
        end = delimiters[i]
        parsed[fields[i]] = data_payload[start:end].decode(
            "ISO-8859-1", errors="replace"
        )

    # Build the address from component fields
    address_parts = []
    for key in ("house", "street", "landmark", "location", "vtc",
                "subdistrict", "district", "postoffice", "state", "pincode"):
        val = parsed.get(key, "").strip()
        if val:
            address_parts.append(val)
    address = ", ".join(address_parts)

    # Extract last 4 digits of Aadhaar from referenceid
    ref_id = parsed.get("referenceid", "")
    aadhaar_last4 = ref_id[:4] if len(ref_id) >= 4 else ref_id

    return {
        "name": parsed.get("name", "").strip(),
        "dob": _normalize_dob(parsed.get("dob", "")),
        "gender": _normalize_gender(parsed.get("gender", "")),
        "aadhaar_number": aadhaar_last4,
        "address": address,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. verify_signature  — RSA signature verification against UIDAI cert
# ──────────────────────────────────────────────────────────────────────────────

def verify_signature(data_payload: bytes, signature: bytes) -> bool:
    """Verify the RSA digital signature against UIDAI's public certificate.

    Parameters
    ----------
    data_payload : bytes
        Everything in the decompressed QR payload EXCEPT the last 256 bytes.
    signature : bytes
        The last 256 bytes of the decompressed payload.

    Returns
    -------
    bool
        True if signature is valid, False otherwise.

    Raises
    ------
    FileNotFoundError
        If the UIDAI public certificate file is missing.
        This is intentionally NOT caught — a missing cert is a setup error,
        not a verification failure.
    """
    if not _CERT_PATH.exists():
        raise FileNotFoundError(
            f"UIDAI public certificate not found at {_CERT_PATH}. "
            f"Please provide the real certificate file — do NOT use a "
            f"self-signed or placeholder certificate."
        )

    # Import here so the module loads even if cryptography isn't installed
    # (decode_qr and extract_qr_fields don't need it).
    try:
        from cryptography.x509 import load_pem_x509_certificate
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        logger.error("'cryptography' package not installed: %s", exc)
        raise

    # Load certificate
    cert_bytes = _CERT_PATH.read_bytes()
    cert = load_pem_x509_certificate(cert_bytes)
    public_key = cert.public_key()

    # Verify: SHA256withRSA, PKCS1v15 padding
    try:
        public_key.verify(
            signature,
            data_payload,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        logger.warning("Digital signature verification FAILED — signature is invalid.")
        return False
    except Exception as exc:
        # Any other crypto error (wrong key type, malformed cert, etc.)
        # must NOT silently return True.
        logger.error("Signature verification error: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 5. cross_check_fields  — compare QR fields vs OCR fields
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_for_comparison(value: str) -> str:
    """Normalize a string for strict comparison.

    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single spaces
    - Lowercase

    NO fuzzy matching, NO Levenshtein distance, NO similarity thresholds.
    """
    return " ".join(value.strip().lower().split())


def cross_check_fields(qr_fields: dict, ocr_fields: dict) -> dict:
    """Compare each field between QR-extracted and OCR-extracted data.

    Parameters
    ----------
    qr_fields : dict
        Output of ``extract_qr_fields()``.
    ocr_fields : dict
        Output of the OCR teammate's module (same key set).

    Returns
    -------
    dict
        Per-field comparison result::

            {
                "name": {"match": True, "qr": "...", "ocr": "..."},
                "dob":  {"match": False, "qr": "...", "ocr": "..."},
                ...
                "all_match": True/False
            }
    """
    compare_keys = ["name", "dob", "gender", "aadhaar_number", "address"]
    result = {}
    all_match = True

    for key in compare_keys:
        qr_val = qr_fields.get(key, "")
        ocr_val = ocr_fields.get(key, "")

        match = _normalize_for_comparison(qr_val) == _normalize_for_comparison(ocr_val)
        if not match:
            all_match = False

        result[key] = {
            "match": match,
            "qr": qr_val,
            "ocr": ocr_val,
        }

    result["all_match"] = all_match
    return result
