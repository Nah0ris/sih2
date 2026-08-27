import cv2
import pytesseract
import re

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_fields(image):

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to black and white
    _, thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Run OCR
    text = pytesseract.image_to_string(thresh)

    print("\n===== OCR TEXT =====")
    print(text)
    print("====================")

    # Extract Aadhaar number
    aadhaar_match = re.search(
        r"\d{4}\s\d{4}\s\d{4}",
        text
    )

    aadhaar_number = ""

    if aadhaar_match:
        aadhaar_number = aadhaar_match.group().replace(" ", "")

    # Extract DOB
    dob_match = re.search(
        r"\d{2}/\d{2}/\d{4}",
        text
    )

    dob = ""

    if dob_match:
        dob = dob_match.group()

    # Extract gender
    gender = ""

    text_lower = text.lower()

    if "female" in text_lower:
        gender = "Female"
    elif "male" in text_lower:
        gender = "Male"
    elif "other" in text_lower:
        gender = "Other"

    # Extract name
    lines = text.split("\n")

    name = ""

    for line in lines:

        line = line.strip()

        if len(line) < 3:
            continue

        if "government" in line.lower():
            continue

        if "india" in line.lower():
            continue

        if "dob" in line.lower():
            continue

        if any(char.isdigit() for char in line):
            continue

        if line.lower() in ["male", "female", "other"]:
            continue

        name = line
        break

    # Address will be improved later
    address = ""

    return {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaar_number": aadhaar_number,
        "address": address
    }


# Test the OCR module
if __name__ == "__main__":

    image_path = "test_data/genuine_sample.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Image could not be loaded.")
        print("Make sure test_data/genuine_sample.jpg exists.")

    else:

        result = extract_fields(image)

        print("\n===== FINAL RESULT =====")
        print(result)
        print("========================")