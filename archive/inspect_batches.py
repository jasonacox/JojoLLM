#!/usr/bin/python3
"""
Detailed inspection of packed data loader batches
Shows sample content, conversation boundaries, and structure
"""

from simple_packed_loader import create_simple_packed_loaders
from setup_tokenizer import get_extended_tokenizer
import torch

def inspect_batch_samples(show_full_sequences=False, num_sequences=2, max_tokens_per_seq=500):
    """
    Inspect the content of packed data loader batches
    
    Args:
        show_full_sequences: If True, show complete sequences (can be very long)
        num_sequences: Number of sequences from batch to inspect
        max_tokens_per_seq: Maximum tokens to show per sequence if not showing full
    """
    print("🔍 Detailed Packed Data Loader Batch Inspection")
    print("=" * 60)
    
    tokenizer = get_extended_tokenizer()
    
    # Test with chitchat data
    train_file = "data/chitchat-train.jsonl"
    val_file = "data/chitchat-val.jsonl"
    
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=4, block_size=512,  # Smaller block size for easier inspection
        train_batches=2, val_batches=1
    )
    
    # Get special token IDs
    try:
        # Try to get vocab directly if it's a tiktoken tokenizer
        if hasattr(tokenizer, 'encode'):
            endoftext_id = tokenizer.encode("<|endoftext|>")[0] if tokenizer.encode("<|endoftext|>") else None
            im_start_id = tokenizer.encode("<|im_start|>")[0] if tokenizer.encode("<|im_start|>") else None
            im_end_id = tokenizer.encode("<|im_end|>")[0] if tokenizer.encode("<|im_end|>") else None
        else:
            endoftext_id = im_start_id = im_end_id = None
    except:
        endoftext_id = im_start_id = im_end_id = None
    
    print(f"🔑 Special Token IDs:")
    print(f"   <|endoftext|>: {endoftext_id}")
    print(f"   <|im_start|>: {im_start_id}")
    print(f"   <|im_end|>: {im_end_id}")
    print()
    
    # Inspect first batch
    for batch_idx, (X, Y) in enumerate(train_dataset):
        print(f"📊 Batch {batch_idx + 1} Analysis:")
        print(f"   Shape: {X.shape}")
        print(f"   Total tokens: {X.numel():,}")
        print(f"   Non-zero tokens: {(X != 0).sum().item():,}")
        print(f"   Efficiency: {(X != 0).sum().item() / X.numel():.1%}")
        print()
        
        # Inspect individual sequences
        for seq_idx in range(min(num_sequences, X.shape[0])):
            print(f"🧩 Sequence {seq_idx + 1}:")
            sequence = X[seq_idx]
            
            # Remove padding (zeros at the end)
            non_zero_mask = sequence != 0
            if non_zero_mask.any():
                last_non_zero = non_zero_mask.nonzero()[-1].item()
                sequence = sequence[:last_non_zero + 1]
            
            print(f"   Length: {len(sequence)} tokens")
            
            # Find conversation boundaries
            if endoftext_id is not None:
                endoftext_positions = (sequence == endoftext_id).nonzero().squeeze().tolist()
                if isinstance(endoftext_positions, int):
                    endoftext_positions = [endoftext_positions]
                print(f"   Conversation boundaries (<|endoftext|>): {len(endoftext_positions)} found at positions {endoftext_positions[:10]}{'...' if len(endoftext_positions) > 10 else ''}")
            
            # Show token sample or full sequence
            if show_full_sequences:
                tokens_to_show = sequence.tolist()
                text_to_show = tokenizer.decode(tokens_to_show)
            else:
                tokens_to_show = sequence[:max_tokens_per_seq].tolist()
                text_to_show = tokenizer.decode(tokens_to_show)
                if len(sequence) > max_tokens_per_seq:
                    text_to_show += f"\n... [truncated, showing {max_tokens_per_seq}/{len(sequence)} tokens]"
            
            print(f"   Tokens: {tokens_to_show[:20]}{'...' if len(tokens_to_show) > 20 else ''}")
            print(f"   Content:")
            print("   " + "="*50)
            for line in text_to_show.split('\n'):
                print(f"   {line}")
            print("   " + "="*50)
            print()
        
        if batch_idx >= 0:  # Only show first batch for detailed inspection
            break
    
    print("✅ Batch inspection complete!")

def quick_sample():
    """Quick sample showing just the decoded text of one sequence"""
    print("⚡ Quick Sample from Packed Data Loader")
    print("=" * 40)
    
    tokenizer = get_extended_tokenizer()
    
    train_dataset, _ = create_simple_packed_loaders(
        "data/chitchat-train.jsonl", "data/chitchat-val.jsonl", 
        tokenizer, batch_size=2, block_size=512, 
        train_batches=1, val_batches=1
    )
    
    # Get first batch
    X, Y = next(iter(train_dataset))
    
    # Show first sequence
    sequence = X[0]
    # Remove padding
    non_zero_mask = sequence != 0
    if non_zero_mask.any():
        last_non_zero = non_zero_mask.nonzero()[-1].item()
        sequence = sequence[:last_non_zero + 1]
    
    text = tokenizer.decode(sequence.tolist())
    
    print(f"📝 Sample sequence ({len(sequence)} tokens):")
    print("-" * 40)
    print(text)
    print("-" * 40)
    print("✅ This shows how conversations are concatenated with <|endoftext|> separators")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        # Detailed inspection
        inspect_batch_samples(
            show_full_sequences="--full" in sys.argv,
            num_sequences=int(sys.argv[sys.argv.index("--sequences") + 1]) if "--sequences" in sys.argv else 2,
            max_tokens_per_seq=int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 500
        )
    else:
        # Quick sample
        quick_sample()
        print("\nFor detailed inspection, run:")
        print("  python inspect_batches.py --detailed")
        print("  python inspect_batches.py --detailed --full")
        print("  python inspect_batches.py --detailed --sequences 3 --tokens 200")
