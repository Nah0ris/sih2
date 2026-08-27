import cv2
import numpy as np


def generate_ela_heatmap(image: np.ndarray) -> tuple[np.ndarray, float]:
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

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

    diff = cv2.absdiff(image, recompressed)

    average_error_score = float(np.mean(diff))

    scaled_diff = np.clip(
        diff * 15,
        0,
        255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        scaled_diff,
        cv2.COLORMAP_JET
    )

    return heatmap, average_error_score
