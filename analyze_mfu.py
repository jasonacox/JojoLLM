#!/usr/bin/env python3
"""
Comprehensive MFU Analysis and Optimization Recommendations
"""

import json

def analyze_configurations():
    """Analyze different model configurations for MFU optimization"""
    
    print("🚀 MFU Configuration Analysis")
    print("=" * 60)
    
    configs = {
        "current": "configs/story-small.json",
        "efficient": "configs/story-efficient.json"
    }
    
    for name, config_path in configs.items():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Calculate approximate parameters
        model = config['model']
        params = estimate_parameters(model)
        
        print(f"\n📊 {name.upper()} CONFIG ({config_path}):")
        print(f"   Model: {model['n_layer']}L × {model['n_head']}H × {model['n_embd']}D")
        print(f"   Parameters: ~{params/1e6:.1f}M")
        print(f"   Sequence length: {model['block_size']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Gradient accum: {config['training']['gradient_accumulation_steps']}")
        print(f"   Effective batch: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
        
        # Memory estimation
        memory_gb = estimate_memory_usage(model, config['training']['batch_size'])
        print(f"   Estimated memory: ~{memory_gb:.1f}GB")
        
        # MFU prediction
        predicted_mfu = predict_mfu(model, config['training']['batch_size'])
        print(f"   Predicted MFU: ~{predicted_mfu:.1f}%")

def estimate_parameters(model_config):
    """Estimate model parameters"""
    L = model_config['n_layer']
    H = model_config['n_head']
    D = model_config['n_embd']
    V = model_config['vocab_size']
    
    # Transformer parameter calculation
    # Attention: 4 * L * D^2 (Q, K, V, O projections)
    attention_params = 4 * L * D * D
    
    # MLP: 8 * L * D^2 (assuming 4*D hidden size)
    mlp_params = 8 * L * D * D
    
    # Embeddings: 2 * V * D (token + position)
    embedding_params = 2 * V * D
    
    # Layer norms (negligible)
    ln_params = 2 * L * D
    
    total = attention_params + mlp_params + embedding_params + ln_params
    return total

def estimate_memory_usage(model_config, batch_size):
    """Estimate GPU memory usage"""
    params = estimate_parameters(model_config)
    seq_len = model_config['block_size']
    
    # Model weights (bfloat16 = 2 bytes per param)
    model_memory = params * 2
    
    # Activations (rough estimate)
    activation_memory = batch_size * seq_len * model_config['n_embd'] * model_config['n_layer'] * 4
    
    # Optimizer states (AdamW: 2x model size)
    optimizer_memory = params * 4
    
    # Gradients
    gradient_memory = params * 2
    
    total_bytes = model_memory + activation_memory + optimizer_memory + gradient_memory
    return total_bytes / (1024**3)  # Convert to GB

def predict_mfu(model_config, batch_size):
    """Predict MFU based on model size and batch size"""
    params = estimate_parameters(model_config)
    seq_len = model_config['block_size']
    
    # Larger models and batches generally get better MFU
    # This is a rough heuristic based on common patterns
    
    if params < 100e6:  # Small models
        base_mfu = 20
    elif params < 300e6:  # Medium models
        base_mfu = 15
    else:  # Large models
        base_mfu = 10
    
    # Batch size impact
    if batch_size >= 32:
        batch_bonus = 15
    elif batch_size >= 16:
        batch_bonus = 10
    elif batch_size >= 8:
        batch_bonus = 5
    else:
        batch_bonus = 0
    
    # Sequence length impact
    if seq_len >= 2048:
        seq_bonus = 5
    elif seq_len >= 1024:
        seq_bonus = 2
    else:
        seq_bonus = 0
    
    return min(base_mfu + batch_bonus + seq_bonus, 50)

def print_recommendations():
    """Print specific recommendations for improving MFU"""
    
    print(f"\n💡 MFU OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 60)
    
    print(f"\n🎯 For your RTX 3090 (23.6GB):")
    
    print(f"\n1. **Use story-efficient.json** for better MFU:")
    print(f"   - Smaller model: 16L × 12H × 768D (~170M params)")
    print(f"   - Shorter sequences: 512 tokens")
    print(f"   - Larger batches: 16 batch size")
    print(f"   - Expected MFU: ~25-35%")
    
    print(f"\n2. **Optimize current large model:**")
    print(f"   - Use batch_size=6 (max that fits)")
    print(f"   - Increase gradient_accumulation_steps=40")
    print(f"   - Consider gradient checkpointing")
    print(f"   - Expected MFU: ~2-5%")
    
    print(f"\n3. **Alternative approaches:**")
    print(f"   - **Reduce sequence length**: 1024 → 512 (4x memory savings)")
    print(f"   - **Use gradient checkpointing**: Trade compute for memory")
    print(f"   - **Model parallelism**: Split across multiple GPUs")
    print(f"   - **Mixed precision**: Already using bfloat16 ✓")
    
    print(f"\n4. **System optimizations:**")
    print(f"   - Set CUDA_LAUNCH_BLOCKING=0")
    print(f"   - Use persistent data workers")
    print(f"   - Monitor GPU utilization with nvidia-smi")
    
    print(f"\n📈 Expected MFU improvements:")
    print(f"   - Current (24L, BS=6): ~0.1-2%")
    print(f"   - Efficient (16L, BS=16): ~25-35%")
    print(f"   - With optimizations: +5-10% additional")

def print_test_commands():
    """Print commands to test different configurations"""
    
    print(f"\n🧪 TEST COMMANDS:")
    print("=" * 40)
    
    print(f"\n# Test efficient configuration:")
    print(f"python train.py --config configs/story-efficient.json --show_config")
    print(f"python train.py --config configs/story-efficient.json --epochs 1")
    
    print(f"\n# Test current configuration with memory limits:")
    print(f"python train.py --config configs/story-small.json --show_config")
    print(f"python train.py --config configs/story-small.json --epochs 1")
    
    print(f"\n# Compare configurations:")
    print(f"python train.py --config configs/story-efficient.json --show_config > efficient_config.txt")
    print(f"python train.py --config configs/story-small.json --show_config > small_config.txt")
    print(f"diff efficient_config.txt small_config.txt")

if __name__ == "__main__":
    analyze_configurations()
    print_recommendations()
    print_test_commands()
    
    print(f"\n🎯 QUICK START:")
    print(f"For immediate MFU improvement, run:")
    print(f"python train.py --config configs/story-efficient.json")
