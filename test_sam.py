import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --- make sure Python can find the submodule ---
sys.path.append(str(Path(__file__).resolve().parent / "segment-anything"))

from segment_anything import sam_model_registry, SamPredictor

# --- model setup ---
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# --- test image ---
image_path = "data/sample_images/test.jpeg"  # place any jpg in your root folder
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not find {image_path}")

predictor.set_image(image)

# --- test segmentation ---
input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
input_label = np.array([1])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

print("✅ SAM ran successfully!")
print("Generated", len(masks), "masks with scores:", scores)

# visualize one mask
plt.imshow(image[..., ::-1])
plt.imshow(masks[0], alpha=0.5)
plt.title("Test SAM Output")
plt.show()
