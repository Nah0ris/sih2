import cv2
import numpy as np


def generate_ela_heatmap(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Generate an Error Level Analysis (ELA) heatmap and normalized error score (0-100).
    Higher scores indicate suspicious multi-compression or copy-paste artifacts.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    # Recompress image at 90% JPEG quality
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        raise ValueError("Failed to encode image")

    recompressed = cv2.imdecode(
        encoded, cv2.IMREAD_COLOR
    )

    # Compute absolute difference
    diff = cv2.absdiff(image, recompressed)

    # Normalized error score (0 - 100 scale)
    raw_mean = float(np.mean(diff))
    average_error_score = min(raw_mean * 25.0, 100.0)

    # Enhance visual contrast for display
    scaled_diff = np.clip(
        diff * 18,
        0,
        255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        scaled_diff,
        cv2.COLORMAP_JET
    )

    return heatmap, average_error_score
