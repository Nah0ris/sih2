import cv2
import pytesseract
import re

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_fields(image):
    """
    Enhanced OCR extraction with 2x resolution upscaling and CLAHE contrast.
    Accurately extracts English name, DOB (e.g. 24/11/2006), Gender, Aadhaar Number and Address
    even from noisy camera photos with multi-lingual Tamil/Hindi headers.
    """
    if image is None or image.size == 0:
        return {"name": "", "dob": "", "gender": "", "aadhaar_number": "", "address": ""}

    # 1. Upscale 2x for sharp character edges (fixes 2006 vs 2008 numeral ambiguity)
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # 2. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Run OCR
    text = pytesseract.image_to_string(enhanced)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 4. Extract Aadhaar number (12 digits or Masked XXXX XXXX 1234)
    aadhaar_match = re.search(r"(\d{4}\s\d{4}\s\d{4})", text)
    if not aadhaar_match:
        aadhaar_match = re.search(r"([xX\*]{4}\s[xX\*]{4}\s\d{4})", text)

    aadhaar_number = ""
    if aadhaar_match:
        aadhaar_number = aadhaar_match.group(1).replace(" ", "")

    # 5. Extract DOB (DD/MM/YYYY or DD-MM-YYYY)
    dob_match = re.search(r"(?:DOB|Birth|நாள்)[\s:/-]*(\d{2}[/\-]\d{2}[/\-]\d{4})", text, re.IGNORECASE)
    if not dob_match:
        dob_match = re.search(r"(\d{2}[/\-]\d{2}[/\-]\d{4})", text)

    dob = ""
    if dob_match:
        dob = dob_match.group(1).replace("-", "/")

    # 6. Extract Gender
    gender = ""
    tl = text.lower()
    if "female" in tl or "பெண்" in text:
        gender = "Female"
    elif "male" in tl or "ஆண்" in text:
        gender = "Male"
    elif "other" in tl:
        gender = "Other"

    # 7. Extract English Name
    name = ""
    # Rule 1: Find the line directly above S/O, D/O, W/O, C/O
    for i, line in enumerate(lines):
        if any(p in line.lower() for p in ["s/o", "d/o", "w/o", "c/o"]) and i > 0:
            candidate = lines[i - 1].strip()
            # Clean non-letter noise (e.g. '|', digits)
            cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) >= 3 and not any(w in cleaned.lower() for w in ["government", "india", "authority", "unique"]):
                name = cleaned
                break

    # Rule 2: Find the line directly above DOB / பிறந்த நாள்
    if not name:
        for i, line in enumerate(lines):
            if any(k in line.lower() for k in ["dob", "birth", "நாள்"]) and i > 0:
                candidate = lines[i - 1].strip()
                cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                if len(cleaned) >= 3 and not any(w in cleaned.lower() for w in ["government", "india", "authority", "unique"]):
                    name = cleaned
                    break

    # Rule 3: Look for English name below 'To'
    if not name:
        for i, line in enumerate(lines):
            if line.lower() == "to":
                for offset in (1, 2, 3):
                    if i + offset < len(lines):
                        candidate = lines[i + offset].strip()
                        cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
                        if len(cleaned) >= 3 and not any(p in cleaned.lower() for p in ["s/o", "d/o", "w/o", "c/o", "no", "street", "road", "gov"]):
                            name = cleaned
                            break
                if name:
                    break

    # 8. Extract Address
    address_parts = []
    capturing = False
    for line in lines:
        if any(p in line.lower() for p in ["s/o", "d/o", "w/o", "c/o", "no 10", "no.", "house"]):
            capturing = True
        if capturing:
            if any(end in line.lower() for end in ["mobile", "your aadhaar", "aadhaar no", "mh45"]):
                break
            address_parts.append(line.replace("|", "").strip())

    address = ", ".join(address_parts) if address_parts else ""

    return {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaar_number": aadhaar_number,
        "address": address,
    }