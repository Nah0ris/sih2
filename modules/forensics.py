import cv2
import numpy as np


def detect_copy_move(image: np.ndarray) -> tuple[bool, list]:
    """
    Detect possible copy-move manipulation using ORB features.

    Args:
        image: Preprocessed BGR image.

    Returns:
        suspicious: True if suspicious duplicated regions are detected.
        matched_regions: List of coordinate pairs for suspicious matches.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1500)

    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Not enough features to compare
    if descriptors is None or len(keypoints) < 2:
        return False, []

    # Brute-force matcher
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    # Match features against themselves
    matches = matcher.match(
        descriptors,
        descriptors
    )

    matched_regions = []

    # Sort matches by distance
    matches = sorted(
        matches,
        key=lambda m: m.distance
    )

    for match in matches:

        # Ignore self-match
        if match.queryIdx == match.trainIdx:
            continue

        p1 = keypoints[match.queryIdx].pt
        p2 = keypoints[match.trainIdx].pt

        # Calculate distance between matched points
        distance = np.linalg.norm(
            np.array(p1) - np.array(p2)
        )

        # Ignore points that are extremely close
        if distance < 30:
            continue

        # Strong ORB match
        if match.distance < 35:
            matched_regions.append(
                (p1, p2)
            )

    # Remove duplicate pairs
    unique_regions = []

    for region in matched_regions:
        if region not in unique_regions:
            unique_regions.append(region)

    # Require multiple matching points
    suspicious = len(unique_regions) >= 8

    return suspicious, unique_regions


def check_layout_consistency(image: np.ndarray) -> dict:
    """
    Check whether important layout elements are approximately
    located where expected.

    This is a prototype layout check using a standardized
    856 x 540 reference layout.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    # Standard size expected from preprocessing
    expected_width = 856
    expected_height = 540

    # Resize input to standard dimensions
    resized = cv2.resize(
        image,
        (expected_width, expected_height)
    )

    flagged_regions = []

    # Expected regions based on reference layout
    expected_regions = {
        "emblem": (20, 20, 120, 110),
        "government_text": (280, 25, 650, 80),
        "name": (50, 130, 400, 190),
        "dob": (50, 190, 400, 250),
        "gender": (50, 250, 400, 310),
        "aadhaar_number": (50, 310, 500, 370),
        "address": (50, 370, 750, 450),
    }

    # Convert image to grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Identify dark text/objects
    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Check every expected region
    for name, (x1, y1, x2, y2) in expected_regions.items():

        region = binary[y1:y2, x1:x2]

        if region.size == 0:
            flagged_regions.append(name)
            continue

        dark_pixel_ratio = np.mean(
            region > 0
        )

        # Very little content means the element
        # may be missing or shifted.
        if dark_pixel_ratio < 0.005:
            flagged_regions.append(name)

    layout_match = len(flagged_regions) == 0

    return {
        "layout_match": layout_match,
        "flagged_regions": flagged_regions
    }


def detect_photo_tampering(image: np.ndarray) -> dict:
    """
    Detect whether the citizen photograph has been digitally spliced, modified, or swapped.

    Analyzes:
      1. Texture Frequency & Sensor Noise Disparity (Laplacian variance discrepancy)
      2. Local ELA Compression Anomaly (pasted photo vs background document)
      3. Boundary Seam Discontinuity (artificial rectangular paste edges)

    Returns:
      {
        "tampering_detected": bool,
        "confidence_score": float,
        "anomalies": list[str]
      }
    """
    if image is None or image.size == 0:
        return {"tampering_detected": False, "confidence_score": 0.0, "anomalies": []}

    h, w = image.shape[:2]
    # Standard photo position in bottom-left card area
    fy1, fy2 = int(h * 0.74), int(h * 0.90)
    fx1, fx2 = int(w * 0.06), int(w * 0.32)

    face_crop = image[fy1:fy2, fx1:fx2]
    if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        return {"tampering_detected": False, "confidence_score": 0.0, "anomalies": []}

    # Reference card background (top header/text area)
    ref_crop = image[:int(h * 0.35), :]

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
    gray_ref = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY) if ref_crop.ndim == 3 else ref_crop

    # 1. Texture / Sensor Noise Disparity
    face_lap = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    ref_lap = float(cv2.Laplacian(gray_ref, cv2.CV_64F).var()) + 1e-5
    lap_ratio = face_lap / ref_lap

    # 2. Local ELA Compression Anomaly
    _, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image, recompressed)

    face_diff = diff[fy1:fy2, fx1:fx2]
    ref_diff = diff[:int(h * 0.35), :]

    face_ela_mean = float(np.mean(face_diff))
    face_ela_max = float(np.max(face_diff))
    ref_ela_mean = float(np.mean(ref_diff)) + 1e-5
    ela_disparity = face_ela_mean / ref_ela_mean

    # 3. Boundary seam gradient (outer perimeter of photo box)
    border_ela = (
        np.mean(diff[max(0, fy1-2):fy1+2, fx1:fx2]) +
        np.mean(diff[max(0, fy2-2):fy2+2, fx1:fx2]) +
        np.mean(diff[fy1:fy2, max(0, fx1-2):fx1+2]) +
        np.mean(diff[fy1:fy2, max(0, fx2-2):fx2+2])
    ) / 4.0

    anomalies = []
    score = 0.0

    # Flag massive frequency/noise jump (typical of pasted external photos)
    if face_lap > 2500.0 or lap_ratio > 3.0:
        score += 45.0
        anomalies.append(f"Sensor noise/frequency discontinuity in photo box (Laplacian var: {face_lap:.0f})")

    if ela_disparity > 1.35 or face_ela_max >= 20.0:
        score += 35.0
        anomalies.append(f"JPEG recompression anomaly in photo box (Disparity: {ela_disparity:.2f}x, Peak: {face_ela_max:.0f})")

    if border_ela > 2.8:
        score += 25.0
        anomalies.append("Artificial rectangular splicing boundary detected around photo perimeter")

    score = min(score, 100.0)
    tampering_detected = score >= 40.0

    return {
        "tampering_detected": tampering_detected,
        "confidence_score": round(score, 1),
        "anomalies": anomalies,
    }
