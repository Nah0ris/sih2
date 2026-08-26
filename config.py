"""
Shared constants and thresholds for the Aadhaar verification pipeline.

Every module imports from here instead of hardcoding values.
Teammates: add your own constants to the relevant section below.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = PROJECT_ROOT / "assets"
UIDAI_CERT_PATH = ASSETS_DIR / "uidai_public_cert.pem"
TEMPLATE_REF_PATH = ASSETS_DIR / "template_reference" / "reference_aadhaar.jpg"

TEST_DATA_DIR = PROJECT_ROOT / "test_data"

# ---------------------------------------------------------------------------
# Preprocessing thresholds (Person 1)
# ---------------------------------------------------------------------------

BLUR_THRESHOLD = 100.0          # Laplacian variance below this → reject as blurry

# ---------------------------------------------------------------------------
# ELA thresholds (Person 3)
# ---------------------------------------------------------------------------

ELA_THRESHOLD = 40.0            # ELA score above this → flag as suspicious (0-100 scale)

# ---------------------------------------------------------------------------
# QR / Crypto (You)
# ---------------------------------------------------------------------------

QR_SIGNATURE_LEN = 256          # RSA-2048 signature = 256 bytes
QR_HASH_LEN = 32                # SHA-256 hash = 32 bytes

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

# Trust score ranges (for reference — actual logic is in verdict.py)
# TAMPERED:   0 - 20
# SUSPICIOUS: 25 - 60
# GENUINE:    80 - 100
