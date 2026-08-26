import cv2
from modules.forensics import (
    detect_copy_move,
    check_layout_consistency
)


# Load test image
image = cv2.imread("test_data/genuine_sample.jpg")

if image is None:
    raise FileNotFoundError(
        "Could not find test_data/genuine_sample.jpg"
    )


# -----------------------------
# Test 1: Copy-Move Detection
# -----------------------------

suspicious, matched_regions = detect_copy_move(image)

print("=== Copy-Move Detection ===")
print("Suspicious:", suspicious)
print("Number of matched regions:", len(matched_regions))
print("Matched regions:", matched_regions)


# -----------------------------
# Test 2: Layout Consistency
# -----------------------------

layout_result = check_layout_consistency(image)

print("\n=== Layout Consistency ===")
print("Layout Match:", layout_result["layout_match"])
print("Flagged Regions:", layout_result["flagged_regions"])
