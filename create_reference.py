import cv2
import numpy as np
import os


# Reference Aadhaar-like layout for testing only
width = 856
height = 540

# White background
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Header
cv2.putText(
    image,
    "GOVERNMENT OF INDIA",
    (300, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

# Synthetic emblem placeholder
cv2.circle(
    image,
    (70, 65),
    35,
    (0, 0, 0),
    2
)

cv2.putText(
    image,
    "EMBLEM",
    (38, 72),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0, 0, 0),
    1
)

# Field labels
fields = [
    ("Name", 80, 160),
    ("DOB", 80, 220),
    ("Gender", 80, 280),
    ("Aadhaar Number", 80, 340),
    ("Address", 80, 400),
]

for label, x, y in fields:
    cv2.putText(
        image,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

# Save reference
output_dir = "assets/template_reference"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "reference_layout.jpg"
)

cv2.imwrite(output_path, image)

print("Reference image created:")
print(output_path)
print("Size:", image.shape)
