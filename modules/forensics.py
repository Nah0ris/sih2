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
