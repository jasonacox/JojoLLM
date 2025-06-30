#!/usr/bin/python3
"""
Simple script to check for special token IDs in binary files
"""
import os
import sys
import numpy as np
import tiktoken

# Add parent directory to path to import setup_tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup_tokenizer import get_extended_tokenizer

def check_dataset(dataset_name):
    """Check a dataset's binary file for special token IDs"""
    train_bin_path = os.path.join('data', f"{dataset_name}-train.bin")
    
    if not os.path.exists(train_bin_path):
        print(f"Error: Could not find {train_bin_path}")
        return False
    
    # Load data from the dataset
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    
    # Define special token IDs
    special_ids = {
        50256: "<|endoftext|>",
        50257: "<|im_start|>",
        50258: "<|im_end|>"
    }
    
    # Count occurrences
    counts = {id_val: 0 for id_val in special_ids.keys()}
    component_tokens = [27, 91, 320, 62, 9688, 437]  # <, |, im, _, start, end
    component_counts = {token: 0 for token in component_tokens}
    
    for token_id in train_data:
        if token_id in counts:
            counts[token_id] += 1
        if token_id in component_counts:
            component_counts[token_id] += 1
    
    # Print results
    print(f"\nResults for {dataset_name}-train.bin:")
    print(f"Total tokens: {len(train_data):,}")
    
    print("\nSpecial token ID counts:")
    for id_val, name in special_ids.items():
        print(f"  {name} (ID {id_val}): {counts[id_val]:,}")
    
    print("\nBroken component token counts:")
    component_names = ["<", "|", "im", "_", "start", "end"]
    for token, name in zip(component_tokens, component_names):
        print(f"  '{name}' (ID {token}): {component_counts[token]:,}")
    
    # Get a sample of tokens to check format
    sample_size = min(1000, len(train_data))
    sample = train_data[:sample_size]
    
    # Try to find sequences that should be special tokens
    sequences = []
    i = 0
    while i < len(sample) - 6:  # Need at least 7 tokens for <|im_start|>
        if (sample[i] == 27 and sample[i+1] == 91 and sample[i+2] == 320 
                and sample[i+3] == 62 and sample[i+4] == 9688 and sample[i+5] == 91 
                and sample[i+6] == 29):
            # Found a potential <|im_start|>
            sequences.append((i, "<|im_start|>"))
            i += 7
        elif (sample[i] == 27 and sample[i+1] == 91 and sample[i+2] == 320 
                and sample[i+3] == 62 and sample[i+4] == 437 and sample[i+5] == 91 
                and sample[i+6] == 29):
            # Found a potential <|im_end|>
            sequences.append((i, "<|im_end|>"))
            i += 7
        else:
            i += 1
    
    if sequences:
        print("\nFound sequences that should be special tokens:")
        for pos, name in sequences[:5]:  # Show first 5 only
            print(f"  {name} at position {pos}")
        if len(sequences) > 5:
            print(f"  ... and {len(sequences) - 5} more")
    
    # Check if this dataset has the issue
    has_issue = (counts[50257] == 0 and counts[50258] == 0 
                and (component_counts[27] > 0 and component_counts[91] > 0))
    
    if has_issue:
        print(f"\n❌ ISSUE DETECTED: {dataset_name} dataset has broken special tokens!")
        print("   The special tokens <|im_start|> and <|im_end|> are encoded as individual components.")
    else:
        print(f"\n✅ {dataset_name} dataset looks good! Special tokens are properly encoded.")
    
    return not has_issue

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check specific dataset
        dataset_name = sys.argv[1]
        check_dataset(dataset_name)
    else:
        # Check all typical datasets
        datasets = ["chat", "chitchat", "dailydialog", "story"]
        results = {}
        
        for dataset in datasets:
            print(f"\n=== Checking {dataset} dataset ===")
            try:
                result = check_dataset(dataset)
                results[dataset] = result
            except Exception as e:
                print(f"Error checking {dataset}: {str(e)}")
                results[dataset] = False
        
        # Print summary
        print("\n=== SUMMARY ===")
        for dataset, result in results.items():
            status = "✅ OK" if result else "❌ ISSUES"
            print(f"{status} - {dataset}")
