import torch
from PIL import Image
from torchvision import transforms
from test_dino import build_dino, args  # your build_dino function and args

# --------------------------
# 1. Build the DINO model
# --------------------------
model, criterion, postprocessors = build_dino(args)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# CRITICAL FIX: Disable intermediate decoder outputs
if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
    model.transformer.decoder.return_intermediate = False
    print("✅ Disabled intermediate decoder outputs")

print(f"✅ DINO model loaded on device: {device}")

# --------------------------
# 2. Load and preprocess the image
# --------------------------
image_path = "data/sample_images/test.jpeg"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((800, 800)),  # DINO typically uses larger inputs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

# --------------------------
# 3. Forward pass
# --------------------------
with torch.no_grad():
    outputs = model(img_tensor)

print("Output type:", type(outputs))
if isinstance(outputs, dict):
    print("Output keys:", outputs.keys())
    if 'pred_boxes' in outputs:
        print("Pred boxes shape:", outputs['pred_boxes'].shape)
        print("Pred logits shape:", outputs['pred_logits'].shape)

# --------------------------
# 4. Post-process outputs (bounding boxes)
# --------------------------
# Use original image size for postprocessing
target_sizes = torch.tensor([[image.height, image.width]]).to(device)

if postprocessors is not None and 'bbox' in postprocessors:
    processed = postprocessors['bbox'](outputs, target_sizes)
    print("\n✅ Processed outputs (bbox):")
    for i, result in enumerate(processed):
        print(f"  Image {i}: {len(result['boxes'])} boxes detected")
        print(f"  Scores shape: {result['scores'].shape}")
        print(f"  Labels shape: {result['labels'].shape}")
else:
    print("No postprocessor for 'bbox' found. Raw outputs shown above.")