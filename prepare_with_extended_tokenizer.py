#!/usr/bin/python3
"""
Jojo LLM Data Preparation Helper

This script provides helper functions to prepare data using
the extended tokenizer with ChatML special tokens. It's designed
to reprocess all conversation datasets using the more token-efficient
special token implementation.

Key benefits:
- Reduces token count for conversational data by ~40%
- Creates more semantically meaningful token boundaries
- Ensures proper handling of special tokens in training data
- Creates debugging files to analyze token boundaries

Usage:
    python prepare_with_extended_tokenizer.py
    
    # This will:
    # 1. Process all .txt datasets in the data/ directory
    # 2. Create corresponding .bin files with tokenized data
    # 3. Generate .tokens.txt and .boundaries.txt debug files

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import os
import numpy as np
import tiktoken
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message, format_system_message

def encode_and_save(text_file, bin_file, tokenizer=None, block_size=1024):
    """
    Encode a text file using the extended tokenizer and save to a binary file.
    
    Args:
        text_file: Path to the input text file
        bin_file: Path to the output binary file
        tokenizer: Optional tokenizer to use (default: extended GPT-2 tokenizer)
        block_size: Block size for encoding (default: 1024)
    """
    # Use the provided tokenizer or get the extended one
    if tokenizer is None:
        tokenizer = get_extended_tokenizer()
    
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode the text (including special tokens)
    tokens = tokenizer.encode(text, allowed_special="all")
    
    # Save the encoded tokens to binary file
    tokens = np.array(tokens, dtype=np.uint16)
    tokens.tofile(bin_file)
    
    # Print some stats
    print(f"Encoded {text_file} to {bin_file}")
    print(f"  - Original text size: {len(text)} characters")
    print(f"  - Encoded to {len(tokens)} tokens")
    print(f"  - Average tokens per character: {len(tokens) / len(text):.2f}")
    
    # Create a text file listing token IDs for debugging
    tokens_debug_file = bin_file + ".tokens.txt"
    with open(tokens_debug_file, 'w') as f:
        f.write(' '.join(map(str, tokens)))
    
    # Create a boundaries file showing logical conversation breaks for debugging
    # This is helpful to see where <|endoftext|> tokens are in the data
    boundaries_file = bin_file + ".boundaries.txt"
    with open(boundaries_file, 'w') as f:
        for i, token in enumerate(tokens):
            if token == 50256:  # <|endoftext|>
                f.write(f"Conversation break at token position {i}\n")
    
    return len(tokens)

def prepare_all_datasets():
    """Prepare all datasets using the extended tokenizer"""
    # Get the extended tokenizer
    tokenizer = get_extended_tokenizer()
    
    # List of dataset paths to process (input text files and output binary files)
    datasets = [
        ("data/chat-train.txt", "data/chat-train.bin"),
        ("data/chat-val.txt", "data/chat-val.bin"),
        ("data/chitchat-train.txt", "data/chitchat-train.bin"),
        ("data/chitchat-val.txt", "data/chitchat-val.bin"),
        ("data/dailydialog-train.txt", "data/dailydialog-train.bin"),
        ("data/dailydialog-val.txt", "data/dailydialog-val.bin"),
        ("data/story-train.bin", "data/story-train.bin"),  # Already processed
        ("data/story-val.bin", "data/story-val.bin"),      # Already processed
    ]
    
    # Process each dataset
    for text_file, bin_file in datasets:
        # Skip if the text file doesn't exist
        if not os.path.exists(text_file):
            print(f"Skipping {text_file} (file not found)")
            continue
            
        # Skip if already processed
        if text_file.endswith('.bin'):
            print(f"Skipping {text_file} (already in binary format)")
            continue
            
        # Process the dataset
        encode_and_save(text_file, bin_file, tokenizer)

if __name__ == "__main__":
    print("\n=== Jojo Data Preparation with Extended Tokenizer ===\n")
    
    # Ask for confirmation
    print("This will re-process all dataset files using the extended tokenizer.")
    print("Existing .bin files will be overwritten.")
    response = input("Do you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        prepare_all_datasets()
        print("\nAll datasets processed successfully!")
    else:
        print("\nOperation cancelled.")
