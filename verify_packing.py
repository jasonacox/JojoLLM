#!/usr/bin/python3
"""
Batch Packing Verification

Specifically checks that batches are fully packed with content
by examining the start and end of each sequence.
"""

from simple_packed_loader import create_simple_packed_loaders
from setup_tokenizer import get_extended_tokenizer

def verify_packing(dataset_name="chitchat", batch_size=4, block_size=1024):
    """Verify that batches are fully packed with content"""
    
    print(f"🔍 Verifying Batch Packing for {dataset_name}")
    print("=" * 60)
    
    tokenizer = get_extended_tokenizer()
    
    # Create data loaders
    train_file = f"data/{dataset_name}-train.jsonl"
    val_file = f"data/{dataset_name}-val.jsonl"
    
    train_dataset, _ = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=batch_size, block_size=block_size,
        train_batches=3, val_batches=1
    )
    
    for batch_idx, (X, Y) in enumerate(train_dataset):
        print(f"\n📦 Batch {batch_idx + 1}:")
        print(f"   Shape: {X.shape}")
        
        batch_total = X.numel()
        batch_effective = (X != 0).sum().item()
        efficiency = batch_effective / batch_total
        print(f"   Efficiency: {efficiency:.1%} ({batch_effective:,}/{batch_total:,} tokens)")
        
        for seq_idx in range(X.shape[0]):
            sequence = X[seq_idx].tolist()
            
            # Get start and end content
            start_tokens = sequence[:30]
            end_tokens = sequence[-30:]
            start_text = tokenizer.decode(start_tokens)
            end_text = tokenizer.decode(end_tokens)
            
            # Check for padding
            zero_count = sequence.count(0)
            
            print(f"\n   🔍 Sequence {seq_idx + 1}:")
            print(f"     START (30 tokens): {start_tokens}")
            print(f"     START text: {repr(start_text)}")
            print(f"     END (30 tokens):   {end_tokens}")
            print(f"     END text:   {repr(end_text)}")
            
            if zero_count > 0:
                first_zero = sequence.index(0)
                print(f"     ⚠️  PADDING: {zero_count} zeros starting at position {first_zero}")
                print(f"     Content ends: {repr(tokenizer.decode(sequence[first_zero-20:first_zero]))}")
            else:
                print(f"     ✅ FULLY PACKED: No padding detected")
        
        if batch_idx >= 2:  # Check 3 batches
            break
    
    print(f"\n🏁 Verification complete!")

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "chitchat"
    verify_packing(dataset)
