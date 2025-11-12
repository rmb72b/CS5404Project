import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2

class SAMModel:
    def __init__(self, checkpoint_path, model_type="vit_b"):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(sam)

    def segment(self, image_path, box):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=box, multimask_output=False
        )
        return masks
