import torch

checkpoint_path = "checkpoint0011_4scale.pth"

print(f"📋 Inspecting checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n✅ Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

if 'args' in checkpoint:
    print("\n📝 Checkpoint args:")
    args = checkpoint['args']
    print(f"  dim_feedforward: {getattr(args, 'dim_feedforward', 'N/A')}")
    print(f"  num_queries: {getattr(args, 'num_queries', 'N/A')}")
    print(f"  two_stage_type: {getattr(args, 'two_stage_type', 'N/A')}")
    print(f"  backbone: {getattr(args, 'backbone', 'N/A')}")
    print(f"  num_feature_levels: {getattr(args, 'num_feature_levels', 'N/A')}")
    print(f"  hidden_dim: {getattr(args, 'hidden_dim', 'N/A')}")
    print(f"  nheads: {getattr(args, 'nheads', 'N/A')}")
    
    print("\n📝 Full args object:")
    print(args)

if 'model' in checkpoint:
    print("\n🔍 Sample model keys (first 20):")
    model_keys = list(checkpoint['model'].keys())[:20]
    for key in model_keys:
        shape = checkpoint['model'][key].shape if hasattr(checkpoint['model'][key], 'shape') else 'N/A'
        print(f"  {key}: {shape}")
    
    # Check specific keys to determine config
    if 'transformer.tgt_embed.weight' in checkpoint['model']:
        tgt_embed_shape = checkpoint['model']['transformer.tgt_embed.weight'].shape
        print(f"\n🎯 Key shape: transformer.tgt_embed.weight = {tgt_embed_shape}")
        print(f"   -> This means num_queries = {tgt_embed_shape[0]}")
    
    if 'transformer.encoder.layers.0.linear1.weight' in checkpoint['model']:
        linear1_shape = checkpoint['model']['transformer.encoder.layers.0.linear1.weight'].shape
        print(f"\n🎯 Key shape: transformer.encoder.layers.0.linear1.weight = {linear1_shape}")
        print(f"   -> This means dim_feedforward = {linear1_shape[0]}")