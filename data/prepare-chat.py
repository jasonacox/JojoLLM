#!/usr/bin/python3
"""
Jojo LLM Data Preparation - Chat Template

This script downloads and prepares conversational datasets formatted with a
Hum        # Download and extract DailyDialog dataset if needed
        zip_path = os.path.join(data_dir, "dailydialog.zip")
        download_if_missing(zip_path, DAILYDIALOG_URL)
        
        # Check if the dialogues_text.txt file exists in the expected location
        dialogues_path = os.path.join(dailydialog_path, "ijcnlp_dailydialog", "dialogues_text.txt")
        if not os.path.exists(dialogues_path):
            extract_zip(zip_path, dailydialog_path)
        
        # Process dialogues
        try:
            print("Processing dialogues...")
            # Path is now properly set to the subdirectoryt chat template suited for training models to engage in
helpful conversations.

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
import re
import argparse
from urllib.parse import urlparse

# Constants
DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"

# Chat format template
HUMAN_PREFIX = "Human: "
ASSISTANT_PREFIX = "Assistant: "
TURN_SEPARATOR = "\n\n"
CONVERSATION_SEPARATOR = "\n\n<|endoftext|>\n\n"

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

def process_dailydialog(dialogue_path):
    """Process the dialogues from the DailyDialog dataset"""
    dialogues = []
    
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        for line in f:
            dialogues.append(line.strip())
    
    return dialogues

def format_chat_template(dialogues):
    """Format dialogues using the Human-Assistant chat template"""
    formatted_data = []
    
    for dialogue in tqdm(dialogues, desc="Formatting dialogues", ncols=80):
        turns = dialogue.split('__eou__')
        # Remove empty turns and the trailing empty string after the last __eou__
        turns = [turn.strip() for turn in turns if turn.strip()]
        
        # Format as a conversation with human-assistant alternating turns
        if len(turns) >= 2:  # Only use dialogues with at least 2 turns
            conversation = []
            for i, turn in enumerate(turns):
                # Fix punctuation spacing issues in all turns
                turn = fix_punctuation_spacing(turn)
                
                # Ensure first turn is always Human
                if i % 2 == 0:
                    # Human turn
                    conversation.append(f"{HUMAN_PREFIX}{turn}")
                else:
                    # Assistant turn
                    # Make sure assistant responses sound helpful and polite
                    turn = refine_assistant_response(turn)
                    conversation.append(f"{ASSISTANT_PREFIX}{turn}")
            
            # Join turns with separator
            formatted_conversation = TURN_SEPARATOR.join(conversation)
            formatted_data.append(formatted_conversation)
    
    return formatted_data

def refine_assistant_response(text):
    """Make the assistant responses more consistent with helpful AI behavior"""
    # Remove any confrontational language
    text = re.sub(r'\b(I disagree|No, that\'s wrong|You\'re wrong|I don\'t want to)\b', 
                  "I understand your perspective", text)
    
    # Ensure responses don't contain inappropriate refusals
    text = re.sub(r'\b(I can\'t help|I won\'t|I refuse)\b', 
                  "I'd be happy to help", text)
    
    # Add more helpful tone for short responses
    if len(text) < 30:
        text = text.rstrip('.') + ". Is there anything else I can help with?"
    
    return text

def fix_punctuation_spacing(text):
    """Fix the odd spacing around punctuation in the DailyDialog dataset"""
    # Remove __eou__ markers if any remain
    text = text.replace('__eou__', '')
    
    # 1. First, standardize ALL apostrophes to the same character - handle both curly (') and straight (') apostrophes
    # This is CRITICAL - we need to convert all apostrophe types to one standard form
    bad_apos = ["`", "’", "‘", "‛", "＇"]
    for apos in bad_apos:
        text = text.replace(apos, "'")

    # Very specific check for the EXACT pattern " ' " (with spaces) and replace directly first
    text = text.replace(" ' ", "'")
    
    # 2. Remove spaces around apostrophes - this handles all cases at once
    # This pattern matches any apostrophe with optional spaces around it
    text = re.sub(r'\s*\'\s*', "'", text)
    
    # 3. Fix contractions specifically to ensure correct formatting
    contractions = {
        r"(\w)'s\b": r"\1's",
        r"(\w)'t\b": r"\1't", 
        r"(\w)'re\b": r"\1're",
        r"(\w)'ve\b": r"\1've",
        r"(\w)'ll\b": r"\1'll",
        r"(\w)'d\b": r"\1'd",
        r"(\w)'m\b": r"\1'm",
    }
    
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    
    # 4. Handle special cases for starting quotes
    text = re.sub(r"(\s|^)'(\w)", r"\1'\2", text)
    
    # 5. Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    
    # 6. Fix double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Clean up any remaining whitespace issues
    text = text.strip()
    
    return text

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare conversational dataset with Human-Assistant chat template')
    parser.add_argument('--source', type=str, default='dailydialog',
                        choices=['dailydialog'], 
                        help='Source dataset to prepare (default: dailydialog)')
    return parser.parse_args()

def main():
    print("\n=== Jojo LLM Chat Template Data Preparation ===\n")
    
    args = parse_args()
    source = args.source
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(data_dir, 'chat')
    ensure_dir(output_base)
    
    if source == 'dailydialog':
        print("Preparing DailyDialog dataset with Human-Assistant chat template...")
        
        # Create directory for dataset files
        dailydialog_path = os.path.join(data_dir, "dailydialog")
        ensure_dir(dailydialog_path)
        
        # Download and extract DailyDialog dataset if needed
        zip_path = os.path.join(data_dir, "dailydialog.zip")
        download_if_missing(zip_path, DAILYDIALOG_URL)
        
        if not os.path.exists(os.path.join(dailydialog_path, "dialogues_text.txt")):
            extract_zip(zip_path, dailydialog_path)
        
        # Process dialogues
        try:
            print("Processing dialogues...")
            dialogues_path = os.path.join(dailydialog_path, "ijcnlp_dailydialog", "dialogues_text.txt")
            dialogues = process_dailydialog(dialogues_path)
            print(f"Loaded {len(dialogues)} dialogues.")
            
            # Format the dialogues with chat template
            formatted_data = format_chat_template(dialogues)
            print(f"Formatted {len(formatted_data)} conversations with Human-Assistant template.")
            
            # Split into train and validation sets
            random.seed(42)  # For reproducibility
            random.shuffle(formatted_data)
            
            split_idx = int(0.9 * len(formatted_data))  # 90% for training, 10% for validation
            train_data = formatted_data[:split_idx]
            val_data = formatted_data[split_idx:]
            
            print(f"Split into {len(train_data)} training and {len(val_data)} validation conversations.")
            
            # Combine all conversations into single text files with conversation separators
            train_text = CONVERSATION_SEPARATOR.join(train_data)
            val_text = CONVERSATION_SEPARATOR.join(val_data)
            
            # Save the raw text files (with -fixed suffix to ensure fresh files)
            train_file = os.path.join(data_dir, "chat-train.txt")
            val_file = os.path.join(data_dir, "chat-val.txt")
            
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
                sys.exit(1)                # Export to bin files for training
            try:
                train_ids = np.array(train_ids, dtype=np.uint16)
                val_ids = np.array(val_ids, dtype=np.uint16)
                
                # Use the original filenames for compatibility with the training script
                train_bin = os.path.join(data_dir, 'chat-train.bin')
                val_bin = os.path.join(data_dir, 'chat-val.bin')
                
                train_ids.tofile(train_bin)
                val_ids.tofile(val_bin)
                
                print(f"Binary files saved to:")
                print(f"  - {train_bin}")
                print(f"  - {val_bin}")
                print("\nPreparation complete! You can now train your model with the chat template dataset.")
                
                # Print usage instructions
                print("\nTo train your model with this chat template dataset, run:")
                print("python train.py --dataset chat")
                
                # Create an example prompt file
                example_dir = os.path.join(os.path.dirname(data_dir), 'examples')
                if not os.path.exists(example_dir):
                    os.makedirs(example_dir)
                
                example_file = os.path.join(example_dir, 'chat_prompt.txt')
                with open(example_file, 'w', encoding='utf-8') as f:
                    f.write(f"{HUMAN_PREFIX}Hello! Can you help me with a question I have?\n\n{ASSISTANT_PREFIX}")
                
                print(f"\nAn example chat prompt has been created at: {example_file}")
                print("Use this with gen.py to test your trained model:")
                print(f"python gen.py models/chat5000.pt --prompt_file examples/chat_prompt.txt")
                
            except Exception as e:
                print(f"Error saving binary files: {str(e)}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error processing dialogues: {str(e)}")
            sys.exit(1)
    
    else:
        print(f"Source '{source}' not supported yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
