import sys, os, torch, types

# Fix imports so local modules are found
repo_root = os.path.join(os.getcwd(), "DINO")
sys.path.append(repo_root)                # DINO/
sys.path.append(os.path.join(repo_root, "models"))  # DINO/models
sys.path.append(os.path.join(repo_root, "util"))    # DINO/util

from models.dino import build_dino

print("CUDA available:", torch.cuda.is_available())

# Full set of arguments for DINO
args = types.SimpleNamespace(
    # Model architecture
    arch='vit_small',
    patch_size=16,
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    dim_feedforward=1024,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_feature_levels=4,
    enc_layers=6,
    dec_layers=6,
    position_embedding='sine',
    pe_temperatureH=10000,
    pe_temperatureW=10000,
    pe_proj_dim=64,
    pe_norm=False,
    matcher_type = 'HungarianMatcher',
    set_cost_class = 1,
    set_cost_bbox = 5,
    set_cost_giou = 2,


    # DINO head / queries / two-stage
    num_queries=300,
    random_refpoints_xy=False,
    fix_refpoints_hw=False,
    two_stage_type='no',
    two_stage_bbox_embed_share=True,
    two_stage_class_embed_share=True,
    decoder_sa_type='ca_label',
    num_patterns=0,
    backbone='resnet50',
    train_backbone=True,                 # add this
    dilation=False,
    return_interm_indices=[0,1,2,3],
    backbone_freeze_keywords=[],

    # Denoising (DN)
    decoder_layer_noise=False,
    unic_layers=6,
    pre_norm=True,
    transformer_activation='relu',
    enc_n_points = 4,
    dec_n_points = 4,
    use_deformable_box_attn = False,
    box_attn_type = "multi_scale",
    add_channel_attention = False,
    add_pos_value = False,
    two_stage_pat_embed = False,
    two_stage_add_query_num = 0,
    two_stage_learn_wh = False,
    two_stage_keep_all_tokens = False,
    dec_layer_number = [300, 1, 1, 1, 1, 1],
    decoder_module_seq = ['sa', 'ca', 'ffn'],
    embed_init_tgt = False,
    query_dim=4,
    use_dn=False,
    dn_number=100,
    dn_box_noise_scale=0.4,
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
    nms_iou_threshold=0.7,
    no_interm_box_loss=False,
    interm_loss_coef=1.0,

    # Misc / optimizer / backbone learning
    lr_backbone=1e-5,
    layer_decay=0.75,
    masks=False,
    frozen_weights=None,
    momentum_teacher=0.996,
    use_fp16=False,
)

import traceback

try:
    model, criterion, postprocessors = build_dino(args)
    print("✅ DINO model built successfully!")
    print(f"Model device: {next(model.parameters()).device}")
except Exception as e:
    print("❌ DINO test failed:")
    traceback.print_exc()
