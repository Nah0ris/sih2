"""
Step 4 verification — test cross_check_fields() and compute_verdict()
against hand-crafted fake input dicts.

Run:  python test_step4.py
"""

import sys
import json

# Ensure modules are importable
sys.path.insert(0, ".")

from modules.qr_crypto import cross_check_fields
from modules.verdict import compute_verdict


def banner(title: str) -> None:
    safe_title = title.encode("ascii", errors="replace").decode("ascii")
    print(f"\n{'='*60}")
    print(f"  {safe_title}")
    print(f"{'='*60}")


def test_cross_check_all_match():
    banner("TEST: cross_check_fields — all fields match")
    qr = {
        "name": "Rajesh Kumar",
        "dob": "15/08/1990",
        "gender": "Male",
        "aadhaar_number": "4321",
        "address": "42 MG Road, Sector 5, Bangalore, Karnataka, 560001",
    }
    ocr = {
        "name": "  Rajesh  Kumar  ",   # extra whitespace → should still match
        "dob": "15/08/1990",
        "gender": "male",              # case difference → should still match
        "aadhaar_number": "4321",
        "address": "42 MG Road, Sector 5, Bangalore, Karnataka, 560001",
    }
    result = cross_check_fields(qr, ocr)
    print(json.dumps(result, indent=2))

    assert result["all_match"] is True, "FAIL: Expected all_match=True"
    assert result["name"]["match"] is True
    assert result["gender"]["match"] is True
    print("[PASS]")


def test_cross_check_name_mismatch():
    banner("TEST: cross_check_fields — name tampered")
    qr = {
        "name": "Rajesh Kumar",
        "dob": "15/08/1990",
        "gender": "Male",
        "aadhaar_number": "4321",
        "address": "42 MG Road, Bangalore",
    }
    ocr = {
        "name": "Suresh Kumar",       # TAMPERED name
        "dob": "15/08/1990",
        "gender": "Male",
        "aadhaar_number": "4321",
        "address": "42 MG Road, Bangalore",
    }
    result = cross_check_fields(qr, ocr)
    print(json.dumps(result, indent=2))

    assert result["all_match"] is False, "FAIL: Expected all_match=False"
    assert result["name"]["match"] is False, "FAIL: Name should not match"
    assert result["dob"]["match"] is True, "FAIL: DOB should match"
    print("[PASS]")


def test_cross_check_multiple_mismatches():
    banner("TEST: cross_check_fields — multiple fields tampered")
    qr = {
        "name": "Priya Sharma",
        "dob": "01/01/2000",
        "gender": "Female",
        "aadhaar_number": "9876",
        "address": "10 Nehru Nagar, Delhi",
    }
    ocr = {
        "name": "Priya Verma",          # tampered
        "dob": "01/01/1995",            # tampered
        "gender": "Female",
        "aadhaar_number": "9876",
        "address": "10 Nehru Nagar, Delhi",
    }
    result = cross_check_fields(qr, ocr)
    print(json.dumps(result, indent=2))

    assert result["all_match"] is False
    assert result["name"]["match"] is False
    assert result["dob"]["match"] is False
    assert result["gender"]["match"] is True
    print("[PASS]")


def test_verdict_genuine():
    banner("TEST: compute_verdict — all checks pass → GENUINE")
    qr_result = {
        "signature_valid": True,
        "fields_cross_check": {
            "name": {"match": True, "qr": "Rajesh", "ocr": "Rajesh"},
            "dob": {"match": True, "qr": "15/08/1990", "ocr": "15/08/1990"},
            "gender": {"match": True, "qr": "Male", "ocr": "Male"},
            "aadhaar_number": {"match": True, "qr": "4321", "ocr": "4321"},
            "address": {"match": True, "qr": "...", "ocr": "..."},
            "all_match": True,
        },
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=10.0,            # low — no manipulation
        copy_move_result={"detected": False},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "GENUINE", f"FAIL: Expected GENUINE, got {verdict['status']}"
    assert verdict["trust_score"] >= 80.0
    print("[PASS]")


def test_verdict_tampered_signature():
    banner("TEST: compute_verdict — invalid signature → TAMPERED")
    qr_result = {
        "signature_valid": False,
        "fields_cross_check": {
            "name": {"match": True, "qr": "Rajesh", "ocr": "Rajesh"},
            "all_match": True,
        },
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=5.0,
        copy_move_result={"detected": False},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "TAMPERED", f"FAIL: Expected TAMPERED, got {verdict['status']}"
    assert verdict["trust_score"] == 0.0
    print("[PASS]")


def test_verdict_tampered_fields():
    banner("TEST: compute_verdict — field mismatch → TAMPERED")
    qr_result = {
        "signature_valid": True,
        "fields_cross_check": {
            "name": {"match": False, "qr": "Rajesh Kumar", "ocr": "Suresh Kumar"},
            "all_match": False,
        },
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=5.0,
        copy_move_result={"detected": False},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "TAMPERED", f"FAIL: Expected TAMPERED, got {verdict['status']}"
    assert verdict["trust_score"] > 0  # sig is valid, just fields mismatch
    print("[PASS]")


def test_verdict_suspicious_ela():
    banner("TEST: compute_verdict — high ELA score → SUSPICIOUS")
    qr_result = {
        "signature_valid": True,
        "fields_cross_check": {"all_match": True},
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=65.0,             # above threshold (40)
        copy_move_result={"detected": False},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "SUSPICIOUS", f"FAIL: Expected SUSPICIOUS, got {verdict['status']}"
    assert 25.0 <= verdict["trust_score"] <= 60.0
    print("[PASS]")


def test_verdict_suspicious_copymove():
    banner("TEST: compute_verdict — copy-move detected → SUSPICIOUS")
    qr_result = {
        "signature_valid": True,
        "fields_cross_check": {"all_match": True},
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=10.0,
        copy_move_result={"detected": True, "details": "region at (100,200)"},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "SUSPICIOUS", f"FAIL: Expected SUSPICIOUS, got {verdict['status']}"
    print("[PASS]")


def test_verdict_qr_unreadable():
    banner("TEST: compute_verdict — QR unreadable → FORENSIC_ONLY (SUSPICIOUS / UNVERIFIED)")
    qr_result = {
        "signature_valid": False,
        "fields_cross_check": None,
        "qr_decode_error": "No QR code found in image",
    }
    verdict = compute_verdict(
        qr_result=qr_result,
        ela_score=20.0,
        copy_move_result={"detected": False},
        layout_result={"valid": True},
    )
    print(json.dumps(verdict, indent=2))

    assert verdict["status"] == "SUSPICIOUS", f"FAIL: Expected SUSPICIOUS, got {verdict['status']}"
    assert verdict["confidence_tier"] == "FORENSIC_ONLY"
    assert verdict["trust_score"] <= 70.0  # capped score
    assert verdict["details"]["qr_readable"] is False
    print("[PASS]")


if __name__ == "__main__":
    tests = [
        test_cross_check_all_match,
        test_cross_check_name_mismatch,
        test_cross_check_multiple_mismatches,
        test_verdict_genuine,
        test_verdict_tampered_signature,
        test_verdict_tampered_fields,
        test_verdict_suspicious_ela,
        test_verdict_suspicious_copymove,
        test_verdict_qr_unreadable,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL]: {e}")
            failed += 1

    banner(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed > 0 else 0)
