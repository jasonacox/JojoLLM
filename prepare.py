import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm

# Free up memory
valid_data = None
train_data = None
train_ids = None
val_ids = None

def ensure_data_dir():
    if not os.path.exists('data'):
        os.makedirs('data')

def download_if_missing(input_file_path, data_url):
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

# URLs
train_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
valid_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# Ensure data directory exists
ensure_data_dir()

# Download the files if they do not exist
train_file = 'data/TinyStoriesV2-GPT4-train.txt'
valid_file = 'data/TinyStoriesV2-GPT4-valid.txt'
download_if_missing(train_file, train_url)
download_if_missing(valid_file, valid_url)

# Load data
print(f"Loading: {train_file}...")
with open(train_file, 'r') as f:
    train_data = f.read()
print(f"Loading: {valid_file}...")
with open(valid_file, 'r') as f:
    valid_data = f.read()

# Encode with tiktoken gpt2
enc = tiktoken.get_encoding("gpt2")
print("Encoding training data...")
train_ids = enc.encode(train_data, allowed_special="all")
print("Encoding validation data...")
val_ids = enc.encode(valid_data, allowed_special="all")
print()
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files for training
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('data/story-train.bin')
val_ids.tofile('data/story-val.bin')