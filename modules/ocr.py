import cv2
import pytesseract
import re

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_fields(image):
    """
    Robust OCR extraction for Indian Aadhaar cards (supports multi-lingual Tamil/Hindi/regional headers).
    """
    if image is None or image.size == 0:
        return {"name": "", "dob": "", "gender": "", "aadhaar_number": "", "address": ""}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Run OCR on clean grayscale
    text = pytesseract.image_to_string(gray)

    # If low text, try thresholded
    if len(text.strip()) < 30:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 1. Extract Aadhaar number (12 digits or Masked XXXX XXXX 1234)
    aadhaar_match = re.search(r"(\d{4}\s\d{4}\s\d{4})", text)
    if not aadhaar_match:
        # Check masked format
        aadhaar_match = re.search(r"([xX\*]{4}\s[xX\*]{4}\s\d{4})", text)

    aadhaar_number = ""
    if aadhaar_match:
        aadhaar_number = aadhaar_match.group(1).replace(" ", "")

    # 2. Extract DOB (DD/MM/YYYY or DD-MM-YYYY or Year-only)
    dob_match = re.search(r"(\d{2}[/\-]\d{2}[/\-]\d{4})", text)
    dob = ""
    if dob_match:
        dob = dob_match.group(1).replace("-", "/")
    else:
        # Check for year-only format
        y_match = re.search(r"(?:DOB|Year of Birth|Birth)[\s:/]+(\d{4})", text, re.IGNORECASE)
        if y_match:
            dob = y_match.group(1)

    # 3. Extract gender
    gender = ""
    text_lower = text.lower()
    if "female" in text_lower or "பெண்" in text:
        gender = "Female"
    elif "male" in text_lower or "ஆண்" in text:
        gender = "Male"
    elif "other" in text_lower:
        gender = "Other"

    # 4. Extract Name
    # In Aadhaar cards, the English name is directly above S/O, D/O, W/O, C/O or below 'To'
    name = ""
    for i, line in enumerate(lines):
        # Rule A: Line directly above S/O, D/O, W/O, C/O
        if any(p in line.lower() for p in ["s/o", "d/o", "w/o", "c/o"]) and i > 0:
            candidate = lines[i - 1].strip()
            # Clean non-alphanumeric noise
            cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
            if len(cleaned) >= 3 and "government" not in cleaned.lower() and "india" not in cleaned.lower():
                name = cleaned
                break

    if not name:
        # Rule B: Line directly above DOB
        for i, line in enumerate(lines):
            if "dob" in line.lower() and i > 0:
                candidate = lines[i - 1].strip()
                cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
                if len(cleaned) >= 3 and "government" not in cleaned.lower() and "india" not in cleaned.lower():
                    name = cleaned
                    break

    if not name:
        # Rule C: First valid English name line after 'To'
        for i, line in enumerate(lines):
            if line.lower() == "to":
                for offset in (1, 2, 3):
                    if i + offset < len(lines):
                        candidate = lines[i + offset].strip()
                        cleaned = re.sub(r"[^A-Za-z\s\.]", "", candidate).strip()
                        if len(cleaned) >= 3 and not any(p in cleaned.lower() for p in ["s/o", "d/o", "w/o", "c/o", "no", "street", "road"]):
                            name = cleaned
                            break
                if name:
                    break

    # 5. Extract Address
    address_lines = []
    capture = False
    for line in lines:
        if any(p in line.lower() for p in ["s/o", "d/o", "w/o", "c/o", "no ", "no.", "house"]):
            capture = True
        if capture:
            if any(end in line.lower() for end in ["pin code", "mobile", "your aadhaar", "aadhaar no"]):
                address_lines.append(line.split(":")[0] if "pin code" not in line.lower() else line)
                break
            address_lines.append(line)

    address = ", ".join(address_lines) if address_lines else ""

    return {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaar_number": aadhaar_number,
        "address": address,
    }