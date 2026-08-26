# AI-Based Fake Identity & Document Screening System

**SIH26188 — Ministry of Home Affairs**

Upload a photo of an Aadhaar card and get an instant verdict: **GENUINE**, **SUSPICIOUS**, or **TAMPERED** — backed by cryptographic QR verification and visual forensics.

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd aadhaar-verifier

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py
```

## Project Structure

```
modules/
  preprocessing.py    — Person 1: perspective correction, CLAHE, blur gate
  ocr.py              — Person 2: field extraction from card text
  ela.py              — Person 3: Error Level Analysis heatmap + score
  forensics.py        — Person 4: copy-move detection + layout/font check
  qr_crypto.py        — Person 5: QR decode, signature verify, field extraction
  verdict.py          — Person 5: combine all outputs into final verdict

ui/
  app.py              — Person 6: Streamlit UI components

config.py             — Shared constants and thresholds
main.py               — Entry point / pipeline integration
```

## How It Works

1. **Preprocess** — correct perspective, enhance contrast, reject blurry images
2. **OCR** — extract name, DOB, gender, address from the card text
3. **QR Crypto** — decode the Secure QR, verify UIDAI digital signature, extract fields
4. **Cross-check** — compare QR fields vs OCR fields (strict match, no fuzzy)
5. **ELA + Forensics** — error-level analysis, copy-move detection, layout check
6. **Verdict** — combine everything into GENUINE / SUSPICIOUS / TAMPERED + trust score

## Requirements

- Python 3.10+
- See `requirements.txt` for packages
- `pyzbar` requires the `zbar` shared library (bundled on Windows via pip)
