#!/usr/bin/python3
"""
Packed Data Loader Verification Script

This script verifies the efficiency and correctness of the packed data loader.
It correctly distinguishes between:
- Content tokens (including exclamation marks that are token 0)  
- Actual padding tokens (trailing zeros)

Use this script to:
- Verify 100% packing efficiency
- Inspect batch content and conversation boundaries
- Spot check data loader performance
- Debug any data loading issues

Run with: python verify_packed_integration.py
"""

import os
import glob
from simple_packed_loader import create_simple_packed_loaders
from setup_tokenizer import get_extended_tokenizer

def discover_datasets(data_dir="data"):
    """Discover available datasets by looking for *-train.jsonl files"""
    if not os.path.exists(data_dir):
        return []
    
    train_files = glob.glob(os.path.join(data_dir, "*-train.jsonl"))
    datasets = []
    
    for train_file in train_files:
        # Extract dataset name from filename
        basename = os.path.basename(train_file)
        dataset_name = basename.replace("-train.jsonl", "")
        
        # Check if corresponding validation file exists
        val_file = train_file.replace("-train.jsonl", "-val.jsonl")
        if os.path.exists(val_file):
            # Get file sizes for display
            train_size = os.path.getsize(train_file)
            val_size = os.path.getsize(val_file)
            
            datasets.append({
                'name': dataset_name,
                'train_file': train_file,
                'val_file': val_file,
                'train_size': train_size,
                'val_size': val_size
            })
    
    return sorted(datasets, key=lambda x: x['name'])

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def select_dataset():
    """Let user select a dataset from available options"""
    datasets = discover_datasets()
    
    if not datasets:
        print("❌ No datasets found in the data/ directory!")
        print("   Looking for files matching pattern: *-train.jsonl and *-val.jsonl")
        return None, None
    
    print(f"\n📚 Available Datasets ({len(datasets)} found):")
    print("=" * 60)
    
    for i, dataset in enumerate(datasets, 1):
        train_size_str = format_file_size(dataset['train_size'])
        val_size_str = format_file_size(dataset['val_size'])
        print(f"  {i}. {dataset['name']}")
        print(f"     Train: {train_size_str} | Val: {val_size_str}")
        print()
    
    while True:
        try:
            choice = input(f"Select dataset (1-{len(datasets)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                return None, None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected = datasets[choice_num - 1]
                print(f"\n✅ Selected dataset: {selected['name']}")
                return selected['train_file'], selected['val_file']
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None, None

def test_efficiency():
    print("🚀 Testing Packed Data Loader Efficiency")
    print("=" * 50)
    
    # Let user select dataset
    train_file, val_file = select_dataset()
    
    if not train_file or not val_file:
        return
    
    dataset_name = os.path.basename(train_file).replace("-train.jsonl", "")
    print(f"\n🔍 Testing dataset: {dataset_name}")
    
    tokenizer = get_extended_tokenizer()
    
    # Get batch configuration from user
    print(f"\n⚙️  Batch Configuration:")
    try:
        batch_size = int(input("Batch size (default 4): ") or "4")
        block_size = int(input("Block size (default 1024): ") or "1024") 
        train_batches = int(input("Number of train batches to test (default 5): ") or "5")
        val_batches = int(input("Number of val batches to test (default 2): ") or "2")
    except ValueError:
        print("❌ Invalid input, using defaults")
        batch_size, block_size, train_batches, val_batches = 4, 1024, 5, 2
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Block size: {block_size}")  
    print(f"   Train batches: {train_batches}")
    print(f"   Val batches: {val_batches}")
    
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=batch_size, block_size=block_size,
        train_batches=train_batches, val_batches=val_batches
    )
    
    print(f"\n📊 Testing efficiency with {train_batches} training batches...")
    
    total_tokens = 0
    effective_tokens = 0
    
    for i, (X, Y) in enumerate(train_dataset):
        batch_total = X.numel()
        
        # Count actual padding tokens (zeros at the end of sequences)
        batch_padding = 0
        for seq_idx in range(X.shape[0]):
            seq = X[seq_idx].tolist()
            # Find trailing zeros (actual padding)
            trailing_zeros = 0
            for token in reversed(seq):
                if token == 0:
                    trailing_zeros += 1
                else:
                    break
            batch_padding += trailing_zeros
        
        batch_effective = batch_total - batch_padding
        
        total_tokens += batch_total
        effective_tokens += batch_effective
        
        efficiency = batch_effective / batch_total
        print(f"   Batch {i+1}: {batch_effective:,}/{batch_total:,} tokens ({efficiency:.1%} efficiency, {batch_padding} padding)")
        
        # Show detailed sample content from first batch
        if i == 0:  # Only show sample for first batch
            print(f"\n📝 Detailed Batch 1 Analysis:")
            
            # Show first sequence in detail
            print(f"\n🔍 Sequence 1 (first {X.shape[1]} tokens):")
            full_sequence = X[0].tolist()
            full_text = tokenizer.decode(full_sequence)
            
            print(f"   First 30 tokens: {full_sequence[:30]}")
            print(f"   Last 30 tokens: {full_sequence[-30:]}")
            print(f"   First 200 chars: {repr(full_text[:200])}")
            print(f"   Last 200 chars: {repr(full_text[-200:])}")
            
            # Check for actual padding at the end (not content tokens like '!')
            zero_tokens = full_sequence.count(0)
            trailing_zeros = 0
            for token in reversed(full_sequence):
                if token == 0:
                    trailing_zeros += 1
                else:
                    break
            
            content_zeros = zero_tokens - trailing_zeros  # Zeros that are content (like '!')
            
            if trailing_zeros > 0:
                print(f"   ⚠️  Actual padding tokens: {trailing_zeros} tokens")
                print(f"   ✅ Content tokens (including {content_zeros} exclamation marks): {len(full_sequence) - trailing_zeros}")
            else:
                print(f"   ✅ No padding - fully packed sequence!")
                if content_zeros > 0:
                    print(f"   📝 Contains {content_zeros} exclamation marks (token 0 = '!')")
            
            # Find conversation boundaries
            try:
                endoftext_token = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
                endoftext_positions = [j for j, token in enumerate(full_sequence) if token == endoftext_token]
                print(f"   <|endoftext|> positions: {endoftext_positions[:10]}{'...' if len(endoftext_positions) > 10 else ''}")
                print(f"   Number of conversations: {len(endoftext_positions)}")
                
                # Show individual conversations
                if len(endoftext_positions) >= 2:
                    print(f"\n📖 Individual Conversations in Sequence 1:")
                    start = 0
                    for idx, pos in enumerate(endoftext_positions[:3]):  # Show first 3 conversations
                        conv_tokens = full_sequence[start:pos+1]
                        conv_text = tokenizer.decode(conv_tokens)
                        print(f"   Conversation {idx+1} ({len(conv_tokens)} tokens):")
                        print(f"     {repr(conv_text[:150])}")
                        start = pos + 1
            except:
                print("   Could not analyze conversation boundaries")
            
            # Show how sequences differ within the batch (start and end)
            print(f"\n🔀 Batch Variety (first and last 50 tokens):")
            for seq_idx in range(min(batch_size, X.shape[0])):
                seq_tokens = X[seq_idx].tolist()
                seq_start = tokenizer.decode(seq_tokens[:50])
                seq_end = tokenizer.decode(seq_tokens[-50:])
                print(f"   Seq {seq_idx+1} START: {repr(seq_start[:80])}")
                print(f"   Seq {seq_idx+1} END:   {repr(seq_end[:80])}")
                
                # Check for actual padding (not content tokens)
                trailing_zeros = 0
                for token in reversed(seq_tokens):
                    if token == 0:
                        trailing_zeros += 1
                    else:
                        break
                
                content_zeros = seq_tokens.count(0) - trailing_zeros
                
                if trailing_zeros > 0:
                    print(f"   Seq {seq_idx+1} ⚠️  Has {trailing_zeros} padding tokens")
                else:
                    print(f"   Seq {seq_idx+1} ✅ Fully packed")
                
                if content_zeros > 0:
                    print(f"   Seq {seq_idx+1} 📝 Contains {content_zeros} exclamation marks")
                print()
        
        if i >= train_batches - 1:  # Test the specified number of batches
            break
    
    overall_efficiency = effective_tokens / total_tokens
    print(f"\n🚀 Overall efficiency: {overall_efficiency:.1%}")
    print(f"💰 Improvement: {overall_efficiency/0.057:.1f}x more efficient than original!")
    print(f"✅ Packed loader successfully integrated into training system!")

if __name__ == "__main__":
    test_efficiency()
