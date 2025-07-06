#!/usr/bin/env python3
"""
Quick MFU test script to find optimal batch size for your hardware
"""

import sys
import os
import torch
import time
import json

# Add current directory to path
sys.path.insert(0, '/home/jason/code/jojo')

from config import Config
from model import GPT, GPTConfig
from utils import MFUCalculator


def test_batch_sizes():
    """Test different batch sizes to find optimal MFU"""
    
    print("🚀 MFU Optimization Test")
    print("=" * 50)
    
    # Load config
    config = Config.from_file('configs/story-small.json')
    
    # Create model
    model_config = GPTConfig(
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        block_size=config.model.block_size,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout
    )
    
    model = GPT(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    # Create MFU calculator
    mfu_calc = MFUCalculator(config.model)
    
    # Test different batch sizes
    test_configs = [
        (8, 60),   # batch_size, grad_accum_steps
        (12, 40),  # current config
        (16, 30),
        (24, 20),  # recommended
        (32, 15),
        (40, 12),
        (48, 10),
    ]
    
    print(f"Hardware: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Peak FLOPS: {mfu_calc.device_peak_flops/1e12:.1f} TFLOPS")
    print(f"Model FLOPS per forward: {mfu_calc.model_flops/1e9:.1f} GFLOPS")
    print()
    
    results = []
    
    for batch_size, grad_accum in test_configs:
        effective_batch = batch_size * grad_accum
        seq_len = config.model.block_size
        
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create dummy data
            x = torch.randint(0, config.model.vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, config.model.vocab_size, (batch_size, seq_len), device=device)
            
            # Warm up
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss.backward()
            
            # Time forward + backward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.time() - start_time
            
            # Calculate MFU
            mfu = mfu_calc.calculate_mfu(batch_size, seq_len, dt)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_pct = memory_used / memory_total * 100
            else:
                memory_used = memory_total = memory_pct = 0
            
            results.append({
                'batch_size': batch_size,
                'grad_accum': grad_accum,
                'effective_batch': effective_batch,
                'mfu': mfu,
                'time_per_batch': dt,
                'memory_gb': memory_used,
                'memory_pct': memory_pct,
                'success': True
            })
            
            status = "✅" if mfu > 30 else "🟡" if mfu > 20 else "🔴"
            print(f"{status} BS={batch_size:2d} GA={grad_accum:2d} EBS={effective_batch:3d} │ "
                  f"MFU={mfu:5.1f}% │ Time={dt*1000:6.1f}ms │ Mem={memory_used:4.1f}GB ({memory_pct:4.1f}%)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💥 BS={batch_size:2d} GA={grad_accum:2d} EBS={effective_batch:3d} │ OUT OF MEMORY")
                results.append({
                    'batch_size': batch_size,
                    'grad_accum': grad_accum,
                    'effective_batch': effective_batch,
                    'success': False,
                    'error': 'OOM'
                })
            else:
                raise e
    
    # Find best configuration
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['mfu'])
        
        print("\n" + "=" * 50)
        print("🏆 OPTIMAL CONFIGURATION:")
        print(f"   batch_size: {best_result['batch_size']}")
        print(f"   gradient_accumulation_steps: {best_result['grad_accum']}")
        print(f"   MFU: {best_result['mfu']:.1f}%")
        print(f"   Memory usage: {best_result['memory_gb']:.1f}GB ({best_result['memory_pct']:.1f}%)")
        
        # Generate optimized config
        optimized_config = config.to_dict()
        optimized_config['training']['batch_size'] = best_result['batch_size']
        optimized_config['training']['gradient_accumulation_steps'] = best_result['grad_accum']
        
        # Save optimized config
        with open('configs/story-small-optimized.json', 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        print(f"\n💾 Saved optimized config to: configs/story-small-optimized.json")
        
        # Show improvement
        current_mfu = next((r['mfu'] for r in successful_results if r['batch_size'] == 12), 0)
        improvement = best_result['mfu'] - current_mfu
        print(f"📈 MFU improvement: {current_mfu:.1f}% → {best_result['mfu']:.1f}% (+{improvement:.1f}%)")
        
        # Optimization hints
        hints = mfu_calc.get_optimization_hints(
            best_result['mfu'], 
            best_result['batch_size'], 
            config.model.block_size
        )
        if hints:
            print(f"\n💡 Additional optimization hints:")
            for hint in hints:
                print(f"   {hint}")


if __name__ == "__main__":
    test_batch_sizes()
