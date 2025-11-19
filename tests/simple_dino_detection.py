import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from test_dino import build_dino, args

# COCO class names
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Build model
print("Building DINO model...")
model, criterion, postprocessors = build_dino(args)

# Load checkpoint
import os
if os.path.exists(args.resume):
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    print("✅ Checkpoint loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
    model.transformer.decoder.return_intermediate = False

print(f"✅ Model ready on {device}")

# Load and preprocess image
image_path = "data/sample_images/test2.jpg"
image = Image.open(image_path).convert("RGB")
orig_width, orig_height = image.size

print(f"Image size: {orig_width}x{orig_height}")

# Standard DINO preprocessing for Swin
transform = transforms.Compose([
    transforms.Resize((800, 1333)),  # Standard DINO size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_tensor = transform(image).unsqueeze(0).to(device)
print(f"Tensor shape: {img_tensor.shape}")

# Inference
print("\nRunning inference...")
with torch.no_grad():
    outputs = model(img_tensor)

print(f"Output keys: {outputs.keys()}")

# Use postprocessor (this handles NMS and coordinate conversion)
orig_target_sizes = torch.tensor([[orig_height, orig_width]]).to(device)
results = postprocessors['bbox'](outputs, orig_target_sizes)

# Get results
boxes = results[0]['boxes'].cpu()
scores = results[0]['scores'].cpu()
labels = results[0]['labels'].cpu()

print(f"\n📊 Postprocessor returned {len(boxes)} detections")
print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")

# Filter by confidence
confidence_threshold = 0.3
keep = scores > confidence_threshold

filtered_boxes = boxes[keep]
filtered_scores = scores[keep]
filtered_labels = labels[keep]

print(f"\n🎯 Detections with confidence > {confidence_threshold}:")
if len(filtered_boxes) == 0:
    print("  No detections above threshold")
    print("\n  Top 10 detections (regardless of threshold):")
    for i in range(min(10, len(boxes))):
        class_name = COCO_CLASSES[labels[i]] if labels[i] < len(COCO_CLASSES) else f"class_{labels[i]}"
        print(f"    {i+1}. {class_name}: {scores[i]:.3f}")
else:
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        print(f"  - {class_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

# Visualize
if len(filtered_boxes) > 0:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        x1, y1, x2, y2 = box.tolist()
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        bbox = draw.textbbox((x1, max(0, y1-25)), text, font=font)
        draw.rectangle(bbox, fill='red')
        draw.text((x1, max(0, y1-25)), text, fill='white', font=font)
    
    output_path = "data/sample_images/test_output.jpeg"
    image.save(output_path)
    print(f"\n✅ Saved to: {output_path}")
else:
    print("\n⚠️  No detections to visualize")