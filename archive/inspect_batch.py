#!/usr/bin/python3
"""
Simple Batch Inspector for Packed Data Loader

Quick tool to inspect what the packed data loader generates.
"""

from simple_packed_loader import create_simple_packed_loaders
from setup_tokenizer import get_extended_tokenizer

def inspect_batch(dataset_name="chitchat", batch_size=4, block_size=512, num_sequences=2):
    """Inspect a single batch from the packed data loader"""
    
    print(f"🔍 Batch Inspector for {dataset_name} dataset")
    print("=" * 60)
    
    tokenizer = get_extended_tokenizer()
    
    # Create data loaders
    train_file = f"data/{dataset_name}-train.jsonl"
    val_file = f"data/{dataset_name}-val.jsonl"
    
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=batch_size, block_size=block_size,
        train_batches=1, val_batches=1
    )
    
    # Get one batch
    X, Y = next(iter(train_dataset))
    
    print(f"📊 Batch Info:")
    print(f"   Shape: {X.shape} (batch_size={X.shape[0]}, block_size={X.shape[1]})")
    print(f"   Total tokens: {X.numel():,}")
    print(f"   Non-zero tokens: {(X != 0).sum().item():,}")
    print(f"   Efficiency: {(X != 0).sum().item() / X.numel():.1%}")
    print()
    
    # Show sequences
    for seq_idx in range(min(num_sequences, X.shape[0])):
        print(f"📝 Sequence {seq_idx + 1}:")
        sequence = X[seq_idx].tolist()
        text = tokenizer.decode(sequence)
        
        # Find conversation boundaries
        try:
            endoftext_token = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
            endoftext_positions = [i for i, token in enumerate(sequence) if token == endoftext_token]
        except:
            endoftext_positions = []
        
        print(f"   Length: {len(sequence)} tokens")
        print(f"   Conversations: {len(endoftext_positions)}")
        print(f"   First 100 chars: {repr(text[:100])}")
        print(f"   Last 100 chars:  {repr(text[-100:])}")
        
        # Check for padding
        zero_count = sequence.count(0)
        if zero_count > 0:
            print(f"   ⚠️  Padding tokens: {zero_count}")
            first_zero = sequence.index(0) if 0 in sequence else -1
            print(f"   First zero at position: {first_zero}")
        else:
            print(f"   ✅ Fully packed - no padding")
        
        if endoftext_positions:
            print(f"   <|endoftext|> at positions: {endoftext_positions[:5]}{'...' if len(endoftext_positions) > 5 else ''}")
            
            # Show first conversation
            if len(endoftext_positions) > 0:
                first_conv_end = endoftext_positions[0]
                first_conv_tokens = sequence[:first_conv_end+1]
                first_conv_text = tokenizer.decode(first_conv_tokens)
                print(f"   First conversation ({len(first_conv_tokens)} tokens):")
                print(f"     {repr(first_conv_text)}")
        print()

def quick_sample(dataset_name="chitchat"):
    """Quick one-liner to see a sample"""
    tokenizer = get_extended_tokenizer()
    train_dataset, _ = create_simple_packed_loaders(
        f"data/{dataset_name}-train.jsonl", f"data/{dataset_name}-val.jsonl", 
        tokenizer, 2, 256, 1, 1
    )
    X, Y = next(iter(train_dataset))
    sample_text = tokenizer.decode(X[0][:100].tolist())
    print(f"Quick sample: {repr(sample_text)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "chitchat"
            quick_sample(dataset)
        else:
            dataset = sys.argv[1]
            inspect_batch(dataset)
    else:
        inspect_batch()
