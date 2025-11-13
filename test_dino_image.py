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
model.aux_loss = False  # ensure no auxiliary outputs to avoid stacking issue

print(f"✅ DINO model loaded on device: {device}")

# --------------------------
# 2. Load and preprocess the image
# --------------------------
image_path = "data/sample_images/test.jpeg"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to model input size
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

# DINO sometimes returns a list if aux_loss was True
# We take only the last output to avoid stack error
if isinstance(outputs, list):
    outputs = outputs[-1]

print("Raw model output shape:", outputs['pred_boxes'].shape)
print("Raw pred_boxes shape:", outputs['pred_boxes'].shape)

# --------------------------
# 4. Post-process outputs (bounding boxes)
# --------------------------
if postprocessors is not None and 'bbox' in postprocessors:
    processed = postprocessors['bbox'](outputs, torch.tensor([img_tensor.shape[-2:]]).to(device))
    print("Processed outputs (bbox):", processed)
else:
    print("No postprocessor for 'bbox' found. Raw outputs shown above.")
