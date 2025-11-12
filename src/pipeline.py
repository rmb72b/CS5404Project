from dino_wrapper import DINOModel
from sam_wrapper import SAMModel

class TwoStageVision:
    def __init__(self, dino_ckpt, sam_ckpt):
        self.detector = DINOModel(dino_ckpt)
        self.segmenter = SAMModel(sam_ckpt)

    def process_image(self, image_path):
        boxes = self.detector.detect(image_path)
        results = []
        for box in boxes:
            mask = self.segmenter.segment(image_path, box)
            results.append((box, mask))
        return results
