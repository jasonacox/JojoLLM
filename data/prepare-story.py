#!/usr/bin/python3
"""
Jojo LLM Data Preparation - TinyStories Dataset

This script downloads and prepares the TinyStories dataset for training.
It downloads the dataset files if they don't exist, converts them to JSONL format
where each line contains a story as a JSON object with "text" field.

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""
import os
import json
import requests
import numpy as np
from tqdm import tqdm
import sys
import re

# Free up memory
valid_data = None
train_data = None


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


def split_stories(text):
    """Split text into individual stories"""
    # Stories are typically separated by double newlines or specific markers
    # TinyStories uses double newlines to separate stories
    stories = []
    
    # Split by double newlines first
    potential_stories = text.split('\n\n')
    
    for story in potential_stories:
        story = story.strip()
        if story and len(story) > 20:  # Filter out very short entries
            # Clean up the story text
            story = re.sub(r'\n+', ' ', story)  # Replace multiple newlines with space
            story = re.sub(r'\s+', ' ', story)  # Replace multiple spaces with single space
            story = story.strip()
            
            if story:
                # Add end of text token
                if not story.endswith('<|endoftext|>'):
                    story += '<|endoftext|>'
                stories.append(story)
    
    return stories


def save_stories_to_jsonl(stories, output_file):
    """Save stories to JSONL format"""
    print(f"Saving {len(stories):,} stories to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for story in tqdm(stories, desc="Writing stories", ncols=80):
            json_obj = {"text": story}
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(stories):,} stories to {output_file}")


def print_dataset_stats(stories, dataset_name):
    """Print statistics about the dataset"""
    if not stories:
        return
    
    story_lengths = [len(story) for story in stories]
    total_chars = sum(story_lengths)
    
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  - Total stories: {len(stories):,}")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Average story length: {total_chars / len(stories):.1f} characters")
    print(f"  - Shortest story: {min(story_lengths)} characters")
    print(f"  - Longest story: {max(story_lengths)} characters")
    
    # Show a sample story
    if stories:
        print(f"\nSample story:")
        sample = stories[len(stories)//2]  # Middle story
        if len(sample) > 200:
            print(f"  {sample[:200]}...")
        else:
            print(f"  {sample}")


def main():
    print("\n=== Jojo LLM Data Preparation - TinyStories Dataset (JSONL Format) ===\n")

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

    # Split into individual stories
    print("\nProcessing training data...")
    train_stories = split_stories(train_data)
    
    print("Processing validation data...")
    valid_stories = split_stories(valid_data)
    
    # Print statistics
    print_dataset_stats(train_stories, "Training")
    print_dataset_stats(valid_stories, "Validation")
    
    # Save to JSONL format
    try:
        train_jsonl = os.path.join(script_dir, 'story-train.jsonl')
        val_jsonl = os.path.join(script_dir, 'story-val.jsonl')
        
        save_stories_to_jsonl(train_stories, train_jsonl)
        save_stories_to_jsonl(valid_stories, val_jsonl)
        
        print(f"\nJSONL files saved:")
        print(f"  - {train_jsonl}")
        print(f"  - {val_jsonl}")
        print(f"\nPreparation complete! You can now train your model with:")
        print(f"  python train.py --dataset story --epochs 1")
        
        # Clean up old binary files if they exist
        old_files = [
            os.path.join(script_dir, 'story-train.bin'),
            os.path.join(script_dir, 'story-val.bin')
        ]
        
        for old_file in old_files:
            if os.path.exists(old_file):
                print(f"\nNote: Found old binary file {old_file}")
                print("The new training system uses JSONL format. You can delete the old .bin files.")
        
    except Exception as e:
        print(f"Error saving JSONL files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
