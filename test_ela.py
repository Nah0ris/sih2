import cv2
from modules.ela import generate_ela_heatmap


image = cv2.imread("test_data/genuine_sample.jpg")

if image is None:
    raise FileNotFoundError(
        "Could not find test_data/genuine_sample.jpg"
    )


heatmap, score = generate_ela_heatmap(image)


print("ELA Score:", score)


cv2.imwrite(
    "test_data/genuine_heatmap.jpg",
    heatmap
)


print("Heatmap saved successfully!")
