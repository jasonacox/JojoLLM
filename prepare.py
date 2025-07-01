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
    python prepare.py
    
    # This will:
    # 1. Process all .txt datasets in the data/ directory
    # 2. Create corresponding .bin files with tokenized data
    # 3. Generate .tokens.txt and .boundaries.txt debug files
    
    # Process a specific dataset:
    python prepare.py --dataset DATASET_NAME
    # Where DATASET_NAME is one of: chat, chitchat, story, knowledge, dictionary

Author: Jason A. Cox
2025 June 30
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
    
    # Check if we need to generate the knowledge dataset
    if not os.path.exists("data/knowledge-train.txt") or not os.path.exists("data/knowledge-val.txt"):
        ensure_knowledge_dataset_exists()
    
    # Check if we need to generate the dictionary dataset
    if not os.path.exists("data/dictionary-train.txt") or not os.path.exists("data/dictionary-val.txt"):
        ensure_dictionary_dataset_exists()
    
    # List of dataset paths to process (input text files and output binary files)
    datasets = [
        ("data/chat-train.txt", "data/chat-train.bin"),
        ("data/chat-val.txt", "data/chat-val.bin"),
        ("data/chitchat-train.txt", "data/chitchat-train.bin"),
        ("data/chitchat-val.txt", "data/chitchat-val.bin"),
        ("data/story-train.bin", "data/story-train.bin"),  # Already processed
        ("data/story-val.bin", "data/story-val.bin"),      # Already processed
        ("data/knowledge-train.txt", "data/knowledge-train.bin"),
        ("data/knowledge-val.txt", "data/knowledge-val.bin"),
        ("data/dictionary-train.txt", "data/dictionary-train.bin"),
        ("data/dictionary-val.txt", "data/dictionary-val.bin"),
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

def process_specific_dataset(dataset_name):
    """Process a specific dataset by name"""
    # Get the extended tokenizer
    tokenizer = get_extended_tokenizer()
    
    # Create a mapping of dataset names to file paths
    dataset_paths = {
        "chat": [("data/chat-train.txt", "data/chat-train.bin"), 
                ("data/chat-val.txt", "data/chat-val.bin")],
        "chitchat": [("data/chitchat-train.txt", "data/chitchat-train.bin"), 
                    ("data/chitchat-val.txt", "data/chitchat-val.bin")],
        "story": [("data/story-train.bin", "data/story-train.bin"), 
                 ("data/story-val.bin", "data/story-val.bin")],
        "knowledge": [("data/knowledge-train.txt", "data/knowledge-train.bin"),
                     ("data/knowledge-val.txt", "data/knowledge-val.bin")],
        "dictionary": [("data/dictionary-train.txt", "data/dictionary-train.bin"),
                      ("data/dictionary-val.txt", "data/dictionary-val.bin")]
    }
    
    # Make sure the dataset name is valid
    if dataset_name not in dataset_paths:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {', '.join(dataset_paths.keys())}")
        return False
    
    # Special handling for knowledge dataset
    if dataset_name == "knowledge" and not os.path.exists("data/knowledge-train.txt"):
        if not ensure_knowledge_dataset_exists():
            print("Failed to generate the knowledge dataset. Please run 'python data/prepare-knowledge.py' manually.")
            return False
            
    # Special handling for dictionary dataset
    if dataset_name == "dictionary" and not os.path.exists("data/dictionary-train.txt"):
        if not ensure_dictionary_dataset_exists():
            print("Failed to generate the dictionary dataset. Please run 'python data/prepare-dictionary.py' manually.")
            return False
        
    # Process the specified dataset
    for text_file, bin_file in dataset_paths[dataset_name]:
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
    
    return True

def ensure_knowledge_dataset_exists():
    """
    Check if the knowledge dataset exists and generate it if needed.
    Returns True if the dataset exists or was successfully generated,
    False otherwise.
    """
    train_file = "data/knowledge-train.txt"
    val_file = "data/knowledge-val.txt"
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        print(f"Knowledge dataset files found.")
        return True
    
    print("Knowledge dataset not found. Running prepare-knowledge.py...")
    print("\nNOTE: This will generate a basic knowledge Q&A dataset.")
    print("For more options, run 'python data/prepare-knowledge.py --help'")
    print("Advanced options include using SQuAD answers directly or using an LLM for reformatting.")
    
    # Run the prepare-knowledge.py script with --use_squad_answers to ensure it works without LLM
    try:
        import subprocess
        result = subprocess.run(["python", "data/prepare-knowledge.py", "--use_squad_answers"], 
                                capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Failed to generate knowledge dataset:")
            print(result.stderr)
            return False
        else:
            print("Knowledge dataset generated successfully!")
            return True
    except Exception as e:
        print(f"An error occurred while generating the knowledge dataset: {e}")
        return False

def ensure_dictionary_dataset_exists():
    """
    Check if the dictionary dataset exists and generate it if needed.
    Returns True if the dataset exists or was successfully generated,
    False otherwise.
    """
    train_file = "data/dictionary-train.txt"
    val_file = "data/dictionary-val.txt"
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        print(f"Dictionary dataset files found.")
        return True
    
    print("Dictionary dataset not found. Running prepare-dictionary.py...")
    print("\nNOTE: This will generate a basic dictionary dataset with word definitions.")
    print("By default, it will process up to 1000 common English words.")
    print("For more options, run 'python data/prepare-dictionary.py --help'")
    
    # Run the prepare-dictionary.py script
    try:
        import subprocess
        result = subprocess.run(["python", "data/prepare-dictionary.py", "--max_words", "100"], 
                                capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Failed to generate dictionary dataset:")
            print(result.stderr)
            return False
        else:
            print("Dictionary dataset generated successfully!")
            return True
    except Exception as e:
        print(f"An error occurred while generating the dictionary dataset: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Jojo Data Preparation with Extended Tokenizer ===\n")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Prepare datasets using the extended tokenizer")
    parser.add_argument("--dataset", type=str, help="Process only the specified dataset (options: chat, chitchat, story, knowledge, dictionary)")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt and process immediately")
    args = parser.parse_args()
    
    # Process a specific dataset if requested
    if args.dataset:
        print(f"Processing dataset: {args.dataset}")
        if process_specific_dataset(args.dataset):
            print(f"\nDataset '{args.dataset}' processed successfully!")
        else:
            print(f"\nFailed to process dataset '{args.dataset}'")
    else:
        # Process all datasets with confirmation
        if not args.force:
            print("This will re-process all dataset files using the extended tokenizer.")
            print("Existing .bin files will be overwritten.")
            response = input("Do you want to continue? (y/n): ")
            
            if response.lower() != 'y':
                print("\nOperation cancelled.")
                exit()
        
        # Ensure knowledge dataset exists before processing
        ensure_knowledge_dataset_exists()
        
        prepare_all_datasets()
        print("\nAll datasets processed successfully!")
