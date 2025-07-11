#!/usr/bin/python3
"""
Jojo LLM Data Preparation - TinyStories Dataset

This script downloads and prepares the TinyStories dataset for training.
It downloads the dataset files if they don't exist, normalizes text characters,
and saves the tokenized data as binary files for efficient loading.

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm
import sys
import unicodedata

# Free up memory
valid_data = None
train_data = None
train_ids = None
val_ids = None

def normalize_text(text):
    """
    Normalize text by replacing problematic characters with standard equivalents.
    Based on analysis of TinyStoriesV2 dataset.
    """
    # Character replacement map based on dataset analysis
    replacements = {
        # Smart quotes to standard quotes
        '"': '"',    # LEFT DOUBLE QUOTATION MARK
        '"': '"',    # RIGHT DOUBLE QUOTATION MARK
        ''': "'",    # LEFT SINGLE QUOTATION MARK
        ''': "'",    # RIGHT SINGLE QUOTATION MARK
        
        # Dashes to standard hyphen
        '–': '-',    # EN DASH
        '—': '-',    # EM DASH
        
        # Ellipsis to three dots
        '…': '...',  # HORIZONTAL ELLIPSIS
        
        # Problematic control characters (found in dataset)
        '\x92': "'", # Unknown control char (likely apostrophe)
        '\x93': '"', # Unknown control char (likely quote)
        '\x94': '"', # Unknown control char (likely quote)
    }
    
    # Apply character replacements
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    # Normalize accented characters (é → e, ñ → n, etc.)
    text = unicodedata.normalize('NFD', text)
    normalized_chars = []
    
    for char in text:
        # Skip combining marks (accents, diacriticals)
        if unicodedata.category(char) != 'Mn':
            normalized_chars.append(char)
    
    text = ''.join(normalized_chars)
    
    # Ensure we only have ASCII characters (safety check)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text

def download_if_missing(input_file_path, data_url):
    """Download a file from data_url if it doesn't exist at input_file_path"""
    if not os.path.exists(input_file_path):
        print(f"Downloading {input_file_path}...")
        response = requests.get(data_url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(input_file_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=input_file_path, ncols=80) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Downloaded {input_file_path}.")

def main():
    print("\n=== Jojo LLM Data Preparation - TinyStories Dataset ===\n")

    # URLs
    train_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    valid_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

    # Get current script directory (should be the data directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define file paths relative to the script directory
    train_file = os.path.join(script_dir, 'TinyStoriesV2-GPT4-train.txt')
    valid_file = os.path.join(script_dir, 'TinyStoriesV2-GPT4-valid.txt')

    # Download the files if they do not exist
    download_if_missing(train_file, train_url)
    download_if_missing(valid_file, valid_url)

    # Load data
    try:
        print(f"Loading: {train_file}...")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = f.read()
        
        print(f"Loading: {valid_file}...")
        with open(valid_file, 'r', encoding='utf-8') as f:
            valid_data = f.read()
            
        # Normalize text characters
        print("Normalizing training data characters...")
        train_data_normalized = normalize_text(train_data)
        print("Normalizing validation data characters...")
        valid_data_normalized = normalize_text(valid_data)
        
        # Check if normalization made changes
        train_changes = len(train_data) - len(train_data_normalized)
        valid_changes = len(valid_data) - len(valid_data_normalized)
        
        if train_changes > 0 or valid_changes > 0:
            print(f"Character normalization complete:")
            print(f"  Training data: {train_changes} characters normalized")
            print(f"  Validation data: {valid_changes} characters normalized")
        else:
            print("No character normalization needed - text is clean!")
            
        # Use normalized data
        train_data = train_data_normalized
        valid_data = valid_data_normalized
    except UnicodeDecodeError:
        print(f"Error: There was an issue with the text encoding. Trying Latin-1 encoding...")
        try:
            with open(train_file, 'r', encoding='latin-1') as f:
                train_data = f.read()
            with open(valid_file, 'r', encoding='latin-1') as f:
                valid_data = f.read()
                
            # Normalize text characters (especially important for Latin-1)
            print("Normalizing training data characters...")
            train_data = normalize_text(train_data)
            print("Normalizing validation data characters...")
            valid_data = normalize_text(valid_data)
            print("Character normalization complete (Latin-1 encoding)")
            
        except Exception as e:
            print(f"Error: Unable to read the data files: {str(e)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Unable to read the data files: {str(e)}")
        sys.exit(1)

    # Encode with tiktoken gpt2
    try:
        enc = tiktoken.get_encoding("gpt2")
        print("Encoding training data...")
        train_ids = enc.encode(train_data, allowed_special="all")
        print("Encoding validation data...")
        val_ids = enc.encode(valid_data, allowed_special="all")
        print()
        print(f"Train data: {len(train_ids):,} tokens")
        print(f"Validation data: {len(val_ids):,} tokens")
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        sys.exit(1)

    # Export to bin files for training
    try:
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        
        train_bin = os.path.join(script_dir, 'story-train.bin')
        val_bin = os.path.join(script_dir, 'story-val.bin')
        
        train_ids.tofile(train_bin)
        val_ids.tofile(val_bin)
        
        print(f"Binary files saved to:")
        print(f"  - {train_bin}")
        print(f"  - {val_bin}")
        print("\nPreparation complete! You can now train your model.")
    except Exception as e:
        print(f"Error saving binary files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
