# pipeline.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from dino_wrapper import run_dino
from sam_wrapper import run_sam

def visualize_results(image_path, boxes, masks, scores=None, labels=None, save_path=None):
    """
    Draws bounding boxes + segmentation masks + labels on the image.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    img_rgb = image[:, :, ::-1]        # BGR → RGB
    overlay = img_rgb.copy()

    # Create different mask colors
    colors = plt.colormaps.get_cmap('hsv')

    for i, mask in enumerate(masks):
        color = np.array(colors(i / len(masks))[:3]) * 255

        # Mask overlay
        overlay[mask > 0] = (
            0.6 * overlay[mask > 0] + 0.4 * color
        ).astype(np.uint8)

        # Bounding box
        x0, y0, x1, y1 = boxes[i].int().tolist()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=2
        )

        # Create label text
        label_text = ""
        if labels is not None:
            label_text = f"{labels[i]}"
        if scores is not None:
            if label_text:
                label_text += f" {scores[i]:.2f}"
            else:
                label_text = f"{scores[i]:.2f}"

        # Draw label background for better readability
        if label_text:
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            cv2.rectangle(
                overlay,
                (x0, y0 - text_height - baseline - 5),
                (x0 + text_width, y0),
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=-1  # Filled rectangle
            )
            cv2.putText(
                overlay,
                label_text,
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )

    # Save or show result
    if save_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert RGB back to BGR for cv2.imwrite
        overlay_bgr = overlay[:, :, ::-1]
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Saved visualization: {save_path}")
    else:
        # Show final result
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("DINO Boxes + SAM Masks")
        plt.show()


def run_pipeline(image_path, confidence_threshold=0.3, max_detections=50):
    print("\n--- Running DINO detection ---")
    boxes, scores, labels = run_dino(image_path)
    print(f"DINO found {len(boxes)} objects (before filtering).")
    
    # Filter by confidence threshold
    confident_indices = scores >= confidence_threshold
    boxes = boxes[confident_indices]
    scores = scores[confident_indices]
    
    # Convert labels list to numpy array for boolean indexing
    labels = np.array(labels)
    labels = labels[confident_indices.numpy()]  # Convert tensor mask to numpy
    labels = labels.tolist()  # Convert back to list
    
    # Limit to top N detections
    if len(boxes) > max_detections:
        top_indices = scores.argsort(descending=True)[:max_detections]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        
        # For labels, convert indices to numpy
        top_indices_np = top_indices.numpy()
        labels = np.array(labels)[top_indices_np].tolist()
    
    print(f"After filtering: {len(boxes)} objects (threshold={confidence_threshold}).")

    print("\n--- Running SAM segmentation ---")
    masks = run_sam(image_path, boxes)
    print(f"SAM generated {len(masks)} masks.")

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"data/sample_outputs/{image_name}_dinosam.jpg"
    visualize_results(image_path, boxes, masks, scores, labels, save_path=output_path)

    return {
        "boxes": boxes,
        "scores": scores,
        "masks": masks,
        "labels": labels
    }

if __name__ == "__main__":
    image = "data/sample_images/test5.jfif"
    # Adjust threshold as needed - start with 0.3 and tune from there
    run_pipeline(image, confidence_threshold=0.3, max_detections=50)