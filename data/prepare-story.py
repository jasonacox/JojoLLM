#!/usr/bin/python3
"""
Jojo LLM Data Preparation - TinyStories Dataset

This script downloads and prepares the TinyStories dataset for training.
It downloads the dataset files if they don't exist, tokenizes the text,
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

# Free up memory
valid_data = None
train_data = None
train_ids = None
val_ids = None

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
    except UnicodeDecodeError:
        print(f"Error: There was an issue with the text encoding. Trying Latin-1 encoding...")
        try:
            with open(train_file, 'r', encoding='latin-1') as f:
                train_data = f.read()
            with open(valid_file, 'r', encoding='latin-1') as f:
                valid_data = f.read()
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
