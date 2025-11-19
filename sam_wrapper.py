# sam_wrapper.py

import torch
import cv2
import numpy as np
from pathlib import Path

# Allow import of the segment-anything module
sam_path = Path(__file__).resolve().parent / "segment-anything"
import sys
sys.path.append(str(sam_path))

from segment_anything import sam_model_registry, SamPredictor

# ----------------------------
# Load SAM once
# ----------------------------
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# ----------------------------
# Run SAM for given boxes
# ----------------------------
def run_sam(image_path, boxes):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    predictor.set_image(image)

    masks = []
    for box in boxes:
        box = np.array(box, dtype=np.float32)

        mask, _, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False
        )

        masks.append(mask[0])

    return masks
