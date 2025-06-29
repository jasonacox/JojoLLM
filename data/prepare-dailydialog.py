#!/usr/bin/python3
"""
Jojo LLM Data Preparation - DailyDialog Dataset

This script downloads and prepares the DailyDialog dataset for training.
DailyDialog is a high-quality multi-turn dialogue dataset that contains
conversations focusing on everyday topics, making it perfect for training
a model to handle casual conversation and small talk.

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm
import zipfile
import json
import sys
import random

# Constants
DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
DAILYDIALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailydialog")
DAILYDIALOG_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailydialog.zip")
DIALOGUES_PATH = os.path.join(DAILYDIALOG_PATH, "ijcnlp_dailydialog", "dialogues_text.txt")

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_if_missing(file_path, url):
    """Download a file if it doesn't exist"""
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
        
    print(f"Downloading {url} to {file_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=file_path, ncols=80) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Downloaded {file_path}.")

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def process_dialogues(dialogue_path):
    """Process the dialogues from the dataset"""
    dialogues = []
    
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        for line in f:
            dialogues.append(line.strip())
    
    return dialogues

def format_for_training(dialogues):
    """Format dialogues for model training"""
    formatted_data = []
    
    for dialogue in tqdm(dialogues, desc="Formatting dialogues", ncols=80):
        turns = dialogue.split('__eou__')
        # Remove empty turns and the trailing empty string after the last __eou__
        turns = [turn.strip() for turn in turns if turn.strip()]
        
        # Format as a conversation
        if len(turns) >= 2:  # Only use dialogues with at least 2 turns
            conversation = ""
            for i, turn in enumerate(turns):
                speaker = "Person A" if i % 2 == 0 else "Person B"
                conversation += f"{speaker}: {turn}\n"
            
            formatted_data.append(conversation.strip())
    
    return formatted_data

def main():
    print("\n=== Jojo LLM Data Preparation - DailyDialog Dataset ===\n")
    
    # Create necessary directories
    ensure_dir(DAILYDIALOG_PATH)
    
    # Download and extract DailyDialog dataset
    download_if_missing(DAILYDIALOG_ZIP, DAILYDIALOG_URL)
    
    if not os.path.exists(os.path.join(DAILYDIALOG_PATH, "dialogues_text.txt")):
        extract_zip(DAILYDIALOG_ZIP, DAILYDIALOG_PATH)
    
    # Process dialogues
    try:
        print("Processing dialogues...")
        dialogues = process_dialogues(DIALOGUES_PATH)
        print(f"Loaded {len(dialogues)} dialogues.")
        
        # Format the dialogues for training
        formatted_data = format_for_training(dialogues)
        print(f"Formatted {len(formatted_data)} conversations for training.")
        
        # Split into train and validation sets
        random.seed(42)  # For reproducibility
        random.shuffle(formatted_data)
        
        split_idx = int(0.9 * len(formatted_data))  # 90% for training, 10% for validation
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        print(f"Split into {len(train_data)} training and {len(val_data)} validation conversations.")
        
        # Combine all conversations into single text files
        train_text = "\n\n<|endoftext|>\n\n".join(train_data)
        val_text = "\n\n<|endoftext|>\n\n".join(val_data)
        
        # Save the raw text files
        train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailydialog-train.txt")
        val_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailydialog-val.txt")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(train_text)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write(val_text)
            
        print(f"Raw text files saved to {train_file} and {val_file}")
        
        # Encode with tiktoken gpt2
        try:
            enc = tiktoken.get_encoding("gpt2")
            print("Encoding training data...")
            train_ids = enc.encode(train_text, allowed_special="all")
            print("Encoding validation data...")
            val_ids = enc.encode(val_text, allowed_special="all")
            
            print(f"Train data: {len(train_ids):,} tokens")
            print(f"Validation data: {len(val_ids):,} tokens")
        except Exception as e:
            print(f"Error during encoding: {str(e)}")
            sys.exit(1)

        # Export to bin files for training
        try:
            train_ids = np.array(train_ids, dtype=np.uint16)
            val_ids = np.array(val_ids, dtype=np.uint16)
            
            train_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dailydialog-train.bin')
            val_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dailydialog-val.bin')
            
            train_ids.tofile(train_bin)
            val_ids.tofile(val_bin)
            
            print(f"Binary files saved to:")
            print(f"  - {train_bin}")
            print(f"  - {val_bin}")
            print("\nPreparation complete! You can now train your model with the DailyDialog dataset.")
            
            # Print usage instructions
            print("\nTo train your model with this conversational dataset, run:")
            print("python train.py --dataset dailydialog")
            
        except Exception as e:
            print(f"Error saving binary files: {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing dialogues: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
