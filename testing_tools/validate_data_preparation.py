#!/usr/bin/python3
"""
Jojo LLM Data Preparation Validation

This script validates that the data preparation process correctly handles special tokens.
It reads a sample of the prepared binary data file, decodes it, and checks if special tokens
are preserved correctly.

Usage:
    python validate_data_preparation.py --dataset chat
    
    # This will:
    # 1. Load a sample of the binary data file
    # 2. Decode it using both standard and extended tokenizer
    # 3. Check if special tokens are preserved intact
    # 4. Report any issues found

Author: Jason A. Cox
2025 June 29
https://github.com/jasonacox/jojo
"""
import os
import sys
import numpy as np
import argparse
import tiktoken

# Add parent directory to path to import setup_tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup_tokenizer import get_extended_tokenizer, SPECIAL_TOKENS

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Validate data preparation for special tokens")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name (e.g., chat, chitchat, dailydialog)")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Number of tokens to sample from the dataset")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset to start sampling from")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed token-by-token information")
    return parser.parse_args()

def check_special_tokens(token_ids, decoded_text):
    """Check if special tokens are preserved correctly"""
    # Define special token IDs
    special_token_ids = {
        "<|im_start|>": 50257, 
        "<|im_end|>": 50258,
        "<|endoftext|>": 50256
    }
    
    # Check if special token IDs are present in the token_ids
    special_tokens_found = {name: 0 for name in special_token_ids.keys()}
    for token_id in token_ids:
        for name, id_val in special_token_ids.items():
            if token_id == id_val:
                special_tokens_found[name] += 1
                
    # Check if special tokens are present in the decoded text
    for name in special_token_ids.keys():
        text_count = decoded_text.count(name)
        if text_count != special_tokens_found[name]:
            return False, f"Mismatch for {name}: Found {special_tokens_found[name]} tokens but {text_count} in text"
    
    return True, "All special tokens matched correctly"

def main():
    """Main function to validate data preparation"""
    args = parse_args()
    
    # Get the dataset path
    data_dir = 'data'
    train_bin_path = os.path.join(data_dir, f"{args.dataset}-train.bin")
    
    if not os.path.exists(train_bin_path):
        print(f"Error: Could not find {train_bin_path}")
        sys.exit(1)
        
    print(f"Validating {args.dataset} dataset...")
    
    # Load some data from the dataset
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    sample_size = min(args.sample_size, len(train_data) - args.offset)
    sample = train_data[args.offset:args.offset + sample_size]
    
    print(f"Loaded {sample_size} tokens from offset {args.offset}")
    
    # Get both tokenizers
    std_enc = tiktoken.get_encoding("gpt2")
    ext_enc = get_extended_tokenizer()
    
    # Convert sample to list for easier handling
    token_ids = [int(token) for token in sample]
    
    # Try decoding with both tokenizers
    print("\nDecoding with standard tokenizer:")
    try:
        std_decoded = std_enc.decode(token_ids)
        print(f"  Success - Length: {len(std_decoded)} chars")
        if args.detailed:
            # Try to decode each token individually
            print("  Individual tokens:")
            for i, token_id in enumerate(token_ids[:20]):  # Show first 20 tokens only
                try:
                    token_text = std_enc.decode([token_id])
                    print(f"    {i}: ID {token_id} = '{token_text}'")
                except:
                    print(f"    {i}: ID {token_id} = <UNKNOWN>")
            if len(token_ids) > 20:
                print(f"    ... {len(token_ids) - 20} more tokens ...")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print("\nDecoding with extended tokenizer:")
    try:
        ext_decoded = ext_enc.decode(token_ids)
        print(f"  Success - Length: {len(ext_decoded)} chars")
        if args.detailed:
            # Try to decode each token individually
            print("  Individual tokens:")
            for i, token_id in enumerate(token_ids[:20]):  # Show first 20 tokens only
                try:
                    token_text = ext_enc.decode([token_id])
                    print(f"    {i}: ID {token_id} = '{token_text}'")
                except:
                    print(f"    {i}: ID {token_id} = <UNKNOWN>")
            if len(token_ids) > 20:
                print(f"    ... {len(token_ids) - 20} more tokens ...")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Check for special tokens
    print("\nChecking for special tokens:")
    
    # With extended tokenizer
    is_valid, message = check_special_tokens(token_ids, ext_decoded)
    if is_valid:
        print(f"  ✅ Extended tokenizer: {message}")
    else:
        print(f"  ❌ Extended tokenizer: {message}")
    
    # Count special token IDs
    special_ids = {
        50256: "<|endoftext|>",
        50257: "<|im_start|>",
        50258: "<|im_end|>"
    }
    
    counts = {id_val: 0 for id_val in special_ids.keys()}
    for token_id in token_ids:
        if token_id in counts:
            counts[token_id] += 1
            
    print("\nSpecial token ID counts:")
    for id_val, name in special_ids.items():
        print(f"  {name} (ID {id_val}): {counts[id_val]}")
    
    # Check for patterns that might indicate broken special tokens
    broken_patterns = [
        "<", "|", "im", "_", "start", "end"
    ]
    
    print("\nChecking for possibly broken special token components:")
    for pattern in broken_patterns:
        try:
            pattern_ids = std_enc.encode(pattern)
            pattern_count = sum(1 for token_id in token_ids if token_id in pattern_ids)
            if pattern_count > 0:
                print(f"  ⚠️ Found {pattern_count} instances of '{pattern}' component")
        except:
            pass
    
    # Print sample of text with correct special tokens
    if not args.detailed and is_valid:
        print("\nSample of correctly tokenized text (first 200 chars):")
        sample_text = ext_decoded[:200]
        sample_text = sample_text.replace("<|im_start|>", "\n<|im_start|>")
        sample_text = sample_text.replace("<|im_end|>", "<|im_end|>\n")
        print(f"  {sample_text}...")

if __name__ == "__main__":
    main()
