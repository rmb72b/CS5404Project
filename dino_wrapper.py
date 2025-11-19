# dino_wrapper.py (MX350-safe version)

import os
import sys
import torch
import types
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Replace the COCO_CLASSES at the top of dino_wrapper.py with this:
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

# Fix repo imports
repo_root = os.path.join(os.getcwd(), "DINO")
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "models"))
sys.path.append(os.path.join(repo_root, "util"))

from models.dino import build_dino

class DictNamespace(types.SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)

# -------- MX350-FRIENDLY CONFIG --------
import types

class DictNamespace(types.SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)

args = DictNamespace(
    # Model architecture
    arch='vit_small',
    patch_size=16,
    hidden_dim=256,
    dropout=0.0,
    nheads=8,
    dim_feedforward=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_feature_levels=4,
    enc_layers=6,
    dec_layers=6,
    position_embedding='sine',
    pe_temperatureH=20,
    pe_temperatureW=20,
    pe_proj_dim=64,
    pe_norm=False,
    matcher_type='HungarianMatcher',
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,

    # DINO head / queries / two-stage
    num_queries=900,
    random_refpoints_xy=False,
    fix_refpoints_hw=-1,
    two_stage_type='standard',
    two_stage_bbox_embed_share=False,
    two_stage_class_embed_share=False,
    decoder_sa_type='sa',
    num_patterns=0,
    backbone='swin_L_384_22k',   # << IMPORTANT
    train_backbone=True,
    dilation=False,
    return_interm_indices=[1,2,3],
    backbone_freeze_keywords=[],
    use_checkpoint=True,

    # DN
    decoder_layer_noise=False,
    unic_layers=6,
    pre_norm=False,
    transformer_activation='relu',
    enc_n_points=4,
    dec_n_points=4,
    use_deformable_box_attn=False,
    box_attn_type="roi_align",
    add_channel_attention=False,
    add_pos_value=False,
    two_stage_pat_embed=0,
    two_stage_add_query_num=0,
    two_stage_learn_wh=False,
    two_stage_keep_all_tokens=False,
    dec_layer_number=None,
    decoder_module_seq=['sa','ca','ffn'],
    embed_init_tgt=True,
    query_dim=4,
    use_dn=True,
    dn_number=100,
    dn_box_noise_scale=1.0,
    dn_label_noise_ratio=0.5,
    dn_labelbook_size=91,
    match_unstable_error=True,
    dec_pred_class_embed_share=True,
    dec_pred_bbox_embed_share=True,

    # Loss / criterion
    num_classes=91,
    device="cuda" if torch.cuda.is_available() else "cpu",
    cls_loss_coef=1.0,
    bbox_loss_coef=5.0,
    giou_loss_coef=2.0,
    mask_loss_coef=1.0,
    dice_loss_coef=1.0,
    aux_loss=True,
    focal_alpha=0.25,
    num_select=300,
    nms_iou_threshold=-1,
    no_interm_box_loss=False,
    interm_loss_coef=1.0,

    # Misc
    lr_backbone=1e-5,
    layer_decay=0.75,
    masks=False,
    frozen_weights=None,
    momentum_teacher=0.996,
    use_fp16=False,

    # Checkpoint
    resume='checkpoint0011_4scale.pth',
    eval=True,
)

print("Building DINO model...")
model, criterion, postprocessors = build_dino(args)

# Load checkpoint safely
if os.path.exists(args.resume):
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("Checkpoint loaded.")

device = args.device
model = model.to(device)
model.eval()

# Disable intermediate returns (like your working test)
if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
    model.transformer.decoder.return_intermediate = False

# -------- Preprocessing used in your good script --------
transform = transforms.Compose([
    transforms.Resize((800, 1333)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------- Run DINO --------
def run_dino(image_path, visualize=True):
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    print("[DINO] Running inference...")
    with torch.no_grad():
        outputs = model(img_tensor)

    orig_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    results = postprocessors["bbox"](outputs, orig_sizes)

    boxes = results[0]["boxes"].cpu()
    scores = results[0]["scores"].cpu()
    labels = results[0]["labels"].cpu()

    # Convert label indices to class names
    label_names = [COCO_CLASSES[idx.item()] if idx.item() < len(COCO_CLASSES) else f"class_{idx.item()}" 
                   for idx in labels]

    # Optional visualization
    if visualize:
        vis = img_pil.copy()
        draw = ImageDraw.Draw(vis)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        # draw top detections
        for box, score, label_name in zip(boxes, scores, label_names):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{label_name}: {score:.2f}"  # Use label_name instead of label.item()
            draw.text((x1, y1 - 20), text, fill="yellow", font=font)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"data/sample_outputs/{image_name}_dino.jpg"
        vis.save(save_path)
        print(f"Saved visualization: {save_path}")

    return boxes, scores, label_names  # Return label_names instead of labels
