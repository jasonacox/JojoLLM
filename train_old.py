#!/usr/bin/python3
"""
Jojo LLM Training Script - Original ONE File Version

This script trains a GPT model on various datasets.

Author: Jason A. Cox
2025 June 29
https://github.com/jasonacox/jojo
"""
import os
import time
import math
import random
import logging
import argparse
import sys
import datetime
import json
import traceback
from pathlib import Path
import tiktoken
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

# Define global variables used throughout the script
train_data = None
val_data = None

# ANSI color codes for better output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
BOLD = '\033[1m'
ENDC = '\033[0m'
UNDERLINE = "\033[4m"

################################# Parameters ##################################
"""
GPT-2 Settings
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
"""

# config
eval_iters = 200 # number of iterations to run evaluation (will be adjusted based on dataset size)
dataset = 'data' # directory where the data is stored
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # [12] number of batches to run in parallel
block_size = 1024 # [32] content window size (tokens)

# model
n_layer = 12     # 4  - layers of 
n_head = 12      # 4  - attention heads
n_embd = 768     # 64 - dimensionality of the embedding vectors
dropout = 0.2   # 0. - for pretraining 0 is good, for finetuning try 0.1+
bias = False    # False - do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4 # 6e-4 max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 60000 # [600000] should be ~= total training steps per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda:1' # examples: 'cpu', 'cuda', or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16'

###############################################################################

# Stats
stat_iter = []  # Will store batch numbers (global batch counter)
stat_loss_train = []
stat_loss_val = []
stat_lr_batch = []  # Store batch numbers for learning rate tracking
stat_lr = []


# Define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model on various datasets')
    parser.add_argument('--dataset', type=str, default='story', 
                        help='Dataset name to use for training (default: story)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for (default: 1)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility (default: 1337)')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='How often to evaluate during an epoch (as a percentage, e.g. 10 means 10 percent intervals)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training progress (as a percentage, e.g. 10 means 10 percent intervals)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to continue training from')
    parser.add_argument('--output_checkpoint', type=str, default=None,
                        help='Custom path to save the output checkpoint (defaults to models/DATASET_epochN.pt)')
    parser.add_argument('--reset_epoch', action='store_true',
                        help='Reset epoch counter to 0 when loading from checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging output')
    return parser.parse_args()

args = parse_args()

# Disable colors if requested
if hasattr(args, 'no_color') and args.no_color:
    GREEN = YELLOW = BLUE = RED = MAGENTA = CYAN = ENDC = BOLD = UNDERLINE = ""

# Set up logging
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=log_level, format=f'{BLUE}%(asctime)s{ENDC} {GREEN}%(levelname)s:{ENDC} %(message)s')
logger = logging.getLogger(__name__)

print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
print(f"{BOLD}{BLUE}║                Jojo LLM Training Program              ║{ENDC}")
print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
print()

# Create a JsonlDataset class to handle JSONL data
class JsonlDataset:
    def __init__(self, jsonl_file, tokenizer):
        """
        Initialize a dataset from a JSONL file where each line contains a JSON object
        with a "text" field containing the conversation data.
        
        Args:
            jsonl_file: Path to the JSONL file
            tokenizer: Tokenizer to use for encoding text
        """
        logger.debug(f"Inside JsonlDataset.__init__ for {jsonl_file}")
        logger.debug(f"Tokenizer type: {type(tokenizer)}")
        
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.data = []
        self.token_count = 0
        self.used_indices = set()  # Track which lines have been used in the current epoch
        self.special_token_stats = {"<|im_start|>": 0, "<|im_end|>": 0, "<|endoftext|>": 0}
        self.reset_epoch()
        
        # Load all data and count total tokens
        print(f"{YELLOW}Loading JSONL data from {jsonl_file}...{ENDC}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "text" in entry:
                        self.data.append(entry)
                        # Count tokens for statistics but don't store them yet
                        try:
                            # Try to tokenize with special tokens allowed
                            tokens = self.tokenizer.encode(entry["text"], allowed_special="all")
                            
                            # Count special tokens for statistics
                            text = entry["text"]
                            for special_token in self.special_token_stats:
                                self.special_token_stats[special_token] += text.count(special_token)
                        except:
                            # Fallback if the extended tokenizer doesn't support allowed_special
                            tokens = self.tokenizer.encode(entry["text"])
                        self.token_count += len(tokens)
                except json.JSONDecodeError:
                    print(f"{RED}Error parsing JSON line in {jsonl_file}{ENDC}")
                except Exception as e:
                    print(f"{RED}Error processing line in {jsonl_file}: {e}{ENDC}")
        
        print(f"{GREEN}Loaded {len(self.data)} conversations with approximately {self.token_count:,} tokens{ENDC}")
        print(f"{GREEN}Using tokenizer: {self.tokenizer.__class__.__name__}{ENDC}")
        
        # Show special token statistics
        self.print_dataset_stats()
    
    def print_dataset_stats(self):
        """Print statistics about the dataset including special token counts"""
        print(f"\n{CYAN}Dataset Statistics for {os.path.basename(self.jsonl_file)}:{ENDC}")
        print(f"  - Conversations: {len(self.data)}")
        print(f"  - Total tokens: {self.token_count:,}")
        print(f"  - Avg tokens per conversation: {self.token_count / len(self.data):.1f}")
        
        # Special token counts
        print(f"  - Special token counts:")
        for token, count in self.special_token_stats.items():
            print(f"    - {token}: {count}")
        print("")
        
    def __len__(self):
        """Return the number of entries in the dataset"""
        return len(self.data)

    def reset_epoch(self):
        """Reset for a new epoch - clear the used indices tracking"""
        logger.debug("Resetting epoch - clearing used indices tracking")
        self.used_indices = set()
    
    def get_batch_data(self, batch_size, block_size, device):
        """
        Get a batch of data ensuring proper epoch coverage.
        Each conversation is used exactly once per epoch before any conversation is reused.
        
        Args:
            batch_size: Number of sequences in the batch
            block_size: Maximum sequence length
            device: Device to put tensors on
            
        Returns:
            x_batch: Input tensors [batch_size, block_size]
            y_batch: Target tensors [batch_size, block_size]
            epoch_complete: Whether we've completed a full epoch
        """
        logger.debug(f"Getting batch data with size={batch_size}, block_size={block_size}")
        x_batch = torch.zeros((batch_size, block_size), dtype=torch.long)
        y_batch = torch.zeros((batch_size, block_size), dtype=torch.long)
        epoch_complete = False
        
        # Process each sequence in the batch
        for b in range(batch_size):
            # Check if we need to reset for a new epoch
            if len(self.used_indices) >= len(self.data):
                print(f"{GREEN}Completed full epoch through {len(self.data)} conversations{ENDC}")
                self.reset_epoch()
                epoch_complete = True
            
            # Get available conversations for this epoch
            available_indices = [i for i in range(len(self.data)) if i not in self.used_indices]
            
            # If no conversations available, we've somehow used all data
            if not available_indices:
                logger.warning("No available conversations - this shouldn't happen")
                self.reset_epoch()
                available_indices = list(range(len(self.data)))
                epoch_complete = True
            
            # Choose a random unused conversation
            conversation_idx = random.choice(available_indices)
            self.used_indices.add(conversation_idx)
            
            # Get the conversation text and tokenize it
            text_to_tokenize = self.data[conversation_idx]["text"]
            try:
                # Try to tokenize with special tokens allowed
                conversation_tokens = self.tokenizer.encode(text_to_tokenize, allowed_special="all")
            except Exception as e:
                # Fallback if the extended tokenizer doesn't support allowed_special
                try:
                    logger.debug(f"Special token encoding failed ({e}), falling back to standard encoding")
                    conversation_tokens = self.tokenizer.encode(text_to_tokenize)
                except Exception as e2:
                    logger.error(f"Error in tokenization: {e2}. Using empty token list.")
                    conversation_tokens = []
            
            # Handle the conversation tokens
            if len(conversation_tokens) == 0:
                logger.warning(f"Empty conversation at index {conversation_idx}, using zeros")
                x_batch[b] = torch.zeros(block_size, dtype=torch.long)
                y_batch[b] = torch.zeros(block_size, dtype=torch.long)
                continue
            
            # If conversation is longer than block_size, take a random slice
            if len(conversation_tokens) > block_size + 1:
                max_start = len(conversation_tokens) - block_size - 1
                start_idx = random.randint(0, max_start)
                tokens = conversation_tokens[start_idx:start_idx + block_size + 1]
            else:
                tokens = conversation_tokens
            
            # Create input/target pairs
            if len(tokens) > block_size:
                x = torch.tensor(tokens[:block_size], dtype=torch.long)
                y = torch.tensor(tokens[1:block_size + 1], dtype=torch.long)
            else:
                # Pad if necessary
                x = torch.tensor(tokens[:-1] if len(tokens) > 1 else tokens, dtype=torch.long)
                y = torch.tensor(tokens[1:] if len(tokens) > 1 else tokens, dtype=torch.long)
                
                # Pad to block_size
                if len(x) < block_size:
                    padding_x = torch.zeros(block_size - len(x), dtype=torch.long)
                    padding_y = torch.zeros(block_size - len(y), dtype=torch.long)
                    x = torch.cat([x, padding_x])
                    y = torch.cat([y, padding_y])
            
            # Add to batch
            x_batch[b] = x
            y_batch[b] = y
        
        # Move to device if specified
        if device is not None:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        
        return x_batch, y_batch, epoch_complete


# Set dataset-specific parameters
dataset_name = args.dataset
print(f"Using dataset: {dataset_name}")

# Set the number of epochs
max_epochs = args.epochs

# Set checkpoint file based on dataset and epochs or use custom path if specified
if args.output_checkpoint:
    checkpoint_file = args.output_checkpoint
else:
    checkpoint_file = f'models/{dataset_name}_epoch{max_epochs}.pt'

# If the output checkpoint exists and is different from the input checkpoint,
# ask user for confirmation before deleting it
if os.path.exists(checkpoint_file) and (args.checkpoint is None or os.path.abspath(checkpoint_file) != os.path.abspath(args.checkpoint)):
    print(f"\n{YELLOW}Warning: Output checkpoint file already exists at:{ENDC}")
    print(f"{BOLD}{checkpoint_file}{ENDC}")
    
    # Ask for confirmation
    while True:
        response = input(f"{BOLD}Delete existing checkpoint file? (y/n): {ENDC}").strip().lower()
        if response == 'y':
            try:
                os.remove(checkpoint_file)
                print(f"{GREEN}Deleted existing file at {checkpoint_file}{ENDC}")
                break
            except Exception as e:
                print(f"{RED}Failed to delete existing checkpoint file: {e}{ENDC}")
                print(f"{YELLOW}Training will continue but may overwrite the file if possible{ENDC}")
                break
        elif response == 'n':
            print(f"{YELLOW}Keeping existing checkpoint file.{ENDC}")
            print(f"{YELLOW}Note: Training will append to or modify the existing file.{ENDC}")
            break
        else:
            print("Please enter 'y' or 'n'.")

# loop counters starting point
epoch = 0
epoch_iter = 0

# capture above settings & parameters to save in model checkpoint
config_keys = [k for k,v in globals().items() 
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))
               and not k in ('args', 'parser')]
config = {k: globals()[k] for k in config_keys} 

# Tokens per batch
tokens_per_batch = gradient_accumulation_steps * batch_size * block_size

# Load the data
data_dir = 'data/'

# Initialize tokenizer for JSONL format
try:
    # Use extended tokenizer with special token support like in prepare-chat.py
    from setup_tokenizer import get_extended_tokenizer
    tokenizer = get_extended_tokenizer()
    print(f"{GREEN}Using extended tokenizer with special token support for JSONL data{ENDC}")
except Exception as e:
    print(f"{RED}Error initializing extended tokenizer: {e}{ENDC}")
    print(f"{RED}Make sure setup_tokenizer.py is properly configured{ENDC}")
    print(f"{YELLOW}Falling back to standard tiktoken cl100k_base...{ENDC}")
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print(f"{YELLOW}Using cl100k_base tokenizer (without special token support){ENDC}")
        print(f"{YELLOW}This may cause issues with special tokens in ChatML format{ENDC}")
    except Exception as e2:
        print(f"{RED}Error initializing fallback tokenizer: {e2}{ENDC}")
        print(f"{RED}Tiktoken encoding library is required for JSONL datasets{ENDC}")
        sys.exit(1)

# Check for JSONL format datasets
train_jsonl_path = os.path.join(data_dir, f'{dataset_name}-train.jsonl')
val_jsonl_path = os.path.join(data_dir, f'{dataset_name}-val.jsonl')

print(f"{GREEN}Loading JSONL format for dataset {dataset_name}{ENDC}")

# Load JSONL datasets
try:
    logger.debug(f"Loading JSONL dataset from {train_jsonl_path}")
    logger.debug(f"Tokenizer type: {type(tokenizer)}")
    train_data = JsonlDataset(train_jsonl_path, tokenizer)
    val_data = JsonlDataset(val_jsonl_path, tokenizer)
    logger.debug(f"Successfully loaded JSONL data. train_data type: {type(train_data)}")
except FileNotFoundError as e:
    print(f"{RED}Error loading JSONL files: {e}{ENDC}")
    print(f"{RED}Make sure both {dataset_name}-train.jsonl and {dataset_name}-val.jsonl exist{ENDC}")
    sys.exit(1)
except Exception as e:
    print(f"{RED}Error processing JSONL data: {e}{ENDC}")
    traceback.print_exc()
    sys.exit(1)

# Calculate total tokens based on dataset size and epochs
tokens_per_epoch = train_data.token_count
total_tokens = tokens_per_epoch * max_epochs

print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
print(f"{BOLD}{BLUE}║                TRAINING CONFIGURATION                 ║{ENDC}")
print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
print(f"{BOLD}Dataset:{ENDC}          {GREEN}{dataset_name}{ENDC}")

# Show epoch information
if epoch > 0:
    print(f"{BOLD}Starting epoch:{ENDC}   {GREEN}{epoch+1}/{max_epochs}{ENDC}")
else:
    print(f"{BOLD}Epochs to run:{ENDC}    {GREEN}{max_epochs}{ENDC}")

# Calculate batches per epoch based on the number of conversations
# Use ceiling division to ensure all conversations are covered
est_batches = max(1, (len(train_data) + batch_size - 1) // batch_size)
print(f"{BOLD}Batches per epoch:{ENDC} {GREEN}{est_batches}{ENDC} (to cover all {len(train_data)} conversations)")

print(f"{BOLD}Batch size:{ENDC}       {GREEN}{batch_size}{ENDC}")
print(f"{BOLD}Block size:{ENDC}       {GREEN}{block_size}{ENDC}")
print(f"{BOLD}Learning rate:{ENDC}    {GREEN}{learning_rate}{ENDC}")
print(f"{BOLD}Tokens per batch:{ENDC} {GREEN}{tokens_per_batch:,}{ENDC}")
print(f"{BOLD}Total tokens:{ENDC}     {GREEN}{total_tokens:,}{ENDC}")
print(f"{BOLD}Device:{ENDC}           {GREEN}{device}{ENDC}")
print(f"{BOLD}Precision:{ENDC}        {GREEN}{dtype}{ENDC}")
if args.checkpoint:
    print(f"{BOLD}Input checkpoint:{ENDC} {GREEN}{args.checkpoint}{ENDC}")
print(f"{BOLD}Output checkpoint:{ENDC}{GREEN}{checkpoint_file}{ENDC}")

# set the random seed
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device selection: auto-detect
if torch.cuda.is_available():
    # CUDA device selection - find device with most free memory
    device_free_memory = []
    max_free_memory = 0
    best_device = 0
    
    print("CUDA devices available:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # Free memory in GB
        mem_total = props.total_memory / (1024 ** 3)  # Total memory in GB
        device_free_memory.append(mem_free)
        
        # Track device with most free memory
        if mem_free > max_free_memory:
            max_free_memory = mem_free
            best_device = i
            
        print(f"  [{i}] {props.name} - Free: {mem_free:.2f} GB / Total: {mem_total:.2f} GB")
    
    # Default to device with most free memory
    selected_device = best_device
    
    # Only ask if there's more than one device
    if torch.cuda.device_count() > 1:
        print(f"{GREEN}Recommended device: [{best_device}] with {max_free_memory:.2f} GB free VRAM{ENDC}")
        try:
            user_input = input(f"Select CUDA device [0-{torch.cuda.device_count()-1}] (default: {best_device}): ")
            # If user enters a value, use it; otherwise, keep the default
            if user_input.strip():
                selected_device = int(user_input)
                if not (0 <= selected_device < torch.cuda.device_count()):
                    print(f"Invalid device index, using recommended device {best_device} instead.")
                    selected_device = best_device
        except Exception:
            print(f"Invalid input, using recommended device {best_device} instead.")
    device = f'cuda:{selected_device}'
    print(f"Using device: {device}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# create an epoch-based data loader for JSONL data
class DataLoader:
    def __init__(self, data, batch_size, block_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.batch_idx = 0  # Initialize batch_idx in constructor
        self.epoch_start_time = time.time()  # Track when the epoch started
        
        # For JSONL datasets, calculate batches needed to see all conversations at least once
        # Each batch uses batch_size conversations, so we need enough batches to cover all data
        self.n_batches = max(1, (len(self.data.data) + batch_size - 1) // batch_size)  # Ceiling division
        print(f"{GREEN}JSONL DataLoader initialized with {self.n_batches} batches per epoch to cover {len(self.data.data)} conversations{ENDC}")

    def __iter__(self):
        logger.debug("DataLoader.__iter__ called - resetting batch_idx to 0")
        self.batch_idx = 0
        # Reset the JSONL dataset for a new epoch
        logger.debug("Resetting JSONL dataset for new epoch")
        self.data.reset_epoch()
        return self

    def __next__(self):
        # Handle JSONL data
        if self.batch_idx >= self.n_batches:
            logger.debug(f"JSONL DataLoader StopIteration at batch {self.batch_idx}/{self.n_batches}")
            raise StopIteration
            
        logger.debug(f"Getting JSONL batch {self.batch_idx+1}/{self.n_batches}")
        self.batch_idx += 1
        # Get a batch from the JSONL dataset - we'll move to device later
        x, y, epoch_complete = self.data.get_batch_data(self.batch_size, self.block_size, None)
        
        # If we completed an epoch and have seen most conversations, we can end early
        if epoch_complete and self.batch_idx >= len(self.data.data) // self.batch_size:
            logger.debug(f"Data epoch completed at batch {self.batch_idx}, ending DataLoader epoch early")
            raise StopIteration
        
        return x, y
        
    def __len__(self):
        return self.n_batches

# get a batch from the data (handles JSONL format only)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    logger.debug(f"Getting {split} batch")
    
    x, y, epoch_complete = data.get_batch_data(batch_size, block_size, device)
    return x, y

# Initialize the model (either from scratch or from checkpoint)
if args.checkpoint:
    print(f"{BOLD}{GREEN}Loading model from checkpoint: {args.checkpoint}{ENDC}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['model_args']
    # Initialize model with args from checkpoint
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Check if model was saved with torch.compile() (keys have "_orig_mod." prefix)
    state_dict = checkpoint['model']
    is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    
    if is_compiled:
        print(f"{YELLOW}Detected compiled model checkpoint. Removing '_orig_mod.' prefix from state dict keys...{ENDC}")
        # Create a new state dict with the prefix removed from each key
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    try:
        # Load model state dict
        model.load_state_dict(state_dict)
        print(f"{GREEN}Model loaded successfully from checkpoint{ENDC}")
    except Exception as e:
        print(f"{RED}Error loading model state dict: {e}{ENDC}")
        print(f"{YELLOW}This could be due to model architecture differences. You may need to train from scratch.{ENDC}")
        sys.exit(1)
        
    # Get epoch and batch counter data from checkpoint
    checkpoint_epoch = checkpoint.get('epoch', 0)
    checkpoint_epoch_iter = checkpoint.get('epoch_iter', 0)
    checkpoint_batch_counter = checkpoint.get('batch_counter', 0)
    
    # Load the batch counter from checkpoint if available
    batch_counter = checkpoint_batch_counter  # This will be used later when batch_counter is needed
    logger.debug(f"Loaded batch_counter={batch_counter} from checkpoint")
    
    if not args.reset_epoch:
        # Continue from checkpoint epoch
        # Check if the model is already trained beyond our target epochs
        print(f"{GREEN}Continuing training from epoch {checkpoint_epoch+1}{ENDC}")
        
        if checkpoint_epoch >= max_epochs:
            print(f"\n{YELLOW}Warning: The loaded model has already been trained for {checkpoint_epoch+1} epochs,{ENDC}")
            print(f"{YELLOW}which is more than or equal to your requested epochs ({max_epochs}).{ENDC}")
            print(f"{YELLOW}This means no additional training will occur unless you increase epochs or use --reset_epoch.{ENDC}\n")
            
            while True:
                response = input(f"{BOLD}How do you want to proceed?{ENDC}\n"
                                f"1) Continue with current settings (no additional training)\n"
                                f"2) Reset epoch counter to 0\n"
                                f"3) Increase epochs\n"
                                f"4) Quit\n"
                                f"Enter choice [1-4]: ").strip()
                
                if response == '1':
                    print(f"{YELLOW}Continuing with current settings (epoch={checkpoint_epoch+1}/{max_epochs}){ENDC}")
                    epoch = checkpoint_epoch
                    epoch_iter = checkpoint_epoch_iter
                    break
                elif response == '2':
                    print(f"{YELLOW}Resetting epoch counter from {checkpoint_epoch+1} to 0{ENDC}")
                    epoch = 0
                    epoch_iter = 0
                    args.reset_epoch = True  # Update flag for consistency
                    break
                elif response == '3':
                    while True:
                        try:
                            new_max = int(input(f"Enter new epochs value (greater than {checkpoint_epoch+1}): "))
                            if new_max > checkpoint_epoch:
                                max_epochs = new_max
                                print(f"{GREEN}Updated max epochs to {max_epochs}{ENDC}")
                                epoch = checkpoint_epoch
                                epoch_iter = checkpoint_epoch_iter
                                # Update checkpoint file path if it's based on epochs
                                if not args.output_checkpoint:
                                    checkpoint_file = f'models/{dataset_name}_epoch{max_epochs}.pt'
                                    print(f"{GREEN}Updated output checkpoint path to: {checkpoint_file}{ENDC}")
                                break
                            else:
                                print(f"{RED}Error: New epochs value must be greater than {checkpoint_epoch+1}{ENDC}")
                        except ValueError:
                            print(f"{RED}Please enter a valid integer{ENDC}")
                    break
                elif response == '4':
                    print(f"{YELLOW}Exiting at user request{ENDC}")
                    sys.exit(0)
                else:
                    print(f"{RED}Invalid choice. Please enter a number between 1 and 4.{ENDC}")
        else:
            # Normal case - continue from checkpoint epoch
            epoch = checkpoint_epoch
            epoch_iter = checkpoint_epoch_iter
            print(f"{GREEN}Continuing from epoch {epoch+1}, batch {epoch_iter}{ENDC}")
    else:
        # Reset epochs as requested
        print(f"{YELLOW}Resetting epoch counter from {checkpoint_epoch+1} to 0{ENDC}")
        epoch = 0
        epoch_iter = 0
    print(f"Training will continue for {max_epochs-epoch} more epochs")
    print(f"Loaded checkpoint: {args.checkpoint}")
else:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    print("Using vocab_size of GPT-2 of 50304 (50257 rounded up for efficiency)")
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=50304, dropout=dropout) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

# report number of parameters
print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
print(f"{BOLD}{BLUE}║             MODEL ARCHITECTURE SUMMARY                ║{ENDC}")
print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
print(f"{BOLD}Total parameters:{ENDC} {GREEN}{model.get_num_params()/1e6:.2f}M{ENDC}")
print(f"\n{BOLD}Key layers:{ENDC}")
for number, (name, param) in enumerate(model.named_parameters()):
    # Only print major layers to avoid overwhelming output
    if '.weight' in name and not 'ln' in name and not 'wpe' in name:
        size_str = f"{np.prod(param.size())/1e6:.2f}M"
        print(f"  {CYAN}[{number}]{ENDC} {name} {YELLOW}({size_str} params){ENDC}")

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler
#scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state if continuing from checkpoint
if args.checkpoint and 'optimizer' in checkpoint:
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"{GREEN}Optimizer state loaded from checkpoint{ENDC}")
    except Exception as e:
        print(f"{YELLOW}Warning: Could not load optimizer state: {e}{ENDC}")

checkpoint = None # free up memory

# compile the model if not using MPS
if device == 'mps':
    print("MPS doesn't support JIT compilation, skipping...")
else:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad() # disable gradient tracking - not needed for backward pass
def estimate_loss():
    out = {}
    model.eval()
    
    # Store the current state of the datasets
    train_used_indices_backup = train_data.used_indices.copy()
    val_used_indices_backup = val_data.used_indices.copy()
    
    for split in ['train', 'val']:
        # Determine appropriate number of evaluation iterations for this split
        data = train_data if split == 'train' else val_data
        # Use a reasonable number of iterations: enough to get a good estimate but not too many
        # For validation, don't use more iterations than we have conversations
        max_reasonable_iters = min(eval_iters, len(data) // batch_size + 1)
        actual_eval_iters = max(10, max_reasonable_iters)  # At least 10 iterations for stability
        
        losses = torch.zeros(actual_eval_iters)
        for k in range(actual_eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    # Restore the state of the datasets
    train_data.used_indices = train_used_indices_backup
    val_data.used_indices = val_used_indices_backup
    
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(epoch_fraction):
    """Get learning rate based on current epoch progress.
    
    Args:
        epoch_fraction: Current epoch as a float (e.g., 2.5 means halfway through epoch 3)
    """
    # Calculate the fraction of total training completed
    total_training_fraction = epoch_fraction / max_epochs
    logger.debug(f"Calculating LR for epoch {epoch_fraction:.2f}/{max_epochs} (fraction: {total_training_fraction:.4f})")
    
    # 1) linear warmup for the first 10% of training
    warmup_fraction = 0.1
    if total_training_fraction < warmup_fraction:
        lr = learning_rate * (total_training_fraction / warmup_fraction)
        logger.debug(f"In warmup phase: LR = {lr:.6f}")
        return lr
    
    # 2) if we're beyond 90% of training, return min learning rate
    cooldown_fraction = 0.9
    if total_training_fraction > cooldown_fraction:
        logger.debug(f"In cooldown phase: LR = {min_lr:.6f}")
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (total_training_fraction - warmup_fraction) / (cooldown_fraction - warmup_fraction)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    lr = min_lr + coeff * (learning_rate - min_lr)
    logger.debug(f"In decay phase: decay_ratio={decay_ratio:.4f}, coeff={coeff:.4f}, LR={lr:.6f}")
    return lr

# Save model to checkpoint file
# Add resume option
resume = False
if os.path.exists(checkpoint_file):
    logger.info(f"Checkpoint {checkpoint_file} found. Use resume option to continue training from checkpoint.")
    # Optionally, load checkpoint here if resume is True

def log_training_start(args):
    """Log the start of a training session to train.log"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = "train.log"
        
        # Collect all command line arguments
        args_str = " ".join([f"--{k}={v}" if v is not True else f"--{k}" 
                           for k, v in vars(args).items() 
                           if v is not False and v is not None])
        
        # Create log entry
        log_entry = (
            f"\n{'='*80}\n"
            f"TRAINING START: {timestamp}\n"
            f"Command: python train.py {args_str}\n"
            f"Dataset: {args.dataset}\n"
            f"Epochs: {args.epochs}\n"
            f"Batch size: {batch_size}, Block size: {block_size}\n"
            f"Learning rate: {learning_rate}\n"
            f"Checkpoint: {args.checkpoint if args.checkpoint else 'None'}\n"
            f"Output: {checkpoint_file}\n"
            f"Data format: JSONL\n"
            f"Device: {device}, Precision: {dtype}\n"
        )
        
        # Append to log file
        with open(log_file, "a") as f:
            f.write(log_entry)
        
        logger.info(f"Training session logged to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log training start: {e}")

def log_training_end(success, duration, train_loss=None, val_loss=None, best_val_loss=None, exception=None):
    """Log the end of a training session to train.log"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = "train.log"
        
        # Format duration as hours:minutes:seconds
        duration_str = str(datetime.timedelta(seconds=int(duration)))
        
        # Create log entry
        log_entry = (
            f"TRAINING END: {timestamp}\n"
            f"Status: {'SUCCESS' if success else 'FAILED'}\n"
            f"Duration: {duration_str}\n"
        )
        
        # Add loss information if available
        if train_loss is not None:
            log_entry += f"Final train loss: {train_loss:.4f}\n"
        if val_loss is not None:
            log_entry += f"Final val loss: {val_loss:.4f}\n"
        if best_val_loss is not None:
            log_entry += f"Best val loss: {best_val_loss:.4f}\n"
        
        # Add exception information if available
        if exception is not None:
            log_entry += f"Exception: {str(exception)}\n"
            
        log_entry += f"Output checkpoint: {checkpoint_file}\n"
        
        # Append to log file
        with open(log_file, "a") as f:
            f.write(log_entry)
        
        logger.info(f"Training results logged to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log training end: {e}")

def save_model(fn):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    
    # Create checkpoint data
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'config': config,
        'dataset': dataset_name,
        'epoch': epoch,
        'epoch_iter': epoch_iter,
        'batch_counter': batch_counter
    }
    # Add timestamp for tracking when the checkpoint was created
    checkpoint['timestamp'] = str(datetime.datetime.now())
    
    # Save the checkpoint with error handling
    try:
        if os.path.exists(fn):
            logger.info(f"{BOLD}{YELLOW}Overwriting existing checkpoint at {fn}{ENDC}")
        else:
            logger.info(f"{BOLD}{GREEN}Saving checkpoint to {fn}{ENDC}")
        
        # First save to a temporary file, then rename to avoid corruption if interrupted
        temp_fn = fn + ".tmp"
        torch.save(checkpoint, temp_fn)
        
        # If we're on Unix, use atomic rename for safer file replacement
        if os.name == 'posix':
            os.replace(temp_fn, fn)
        else:
            # On Windows, remove destination first (not atomic)
            if os.path.exists(fn):
                os.remove(fn)
            os.rename(temp_fn, fn)
            
        logger.info(f"{BOLD}{GREEN}Successfully saved checkpoint to {fn}{ENDC}")
        return True
    except Exception as e:
        logger.error(f"{BOLD}{RED}Failed to save checkpoint: {e}{ENDC}")
        # Try to clean up temp file if it exists
        try:
            if os.path.exists(temp_fn):
                os.remove(temp_fn)
        except:
            pass
        return False

# === Plot loss curves and save as PNG ===
def plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, checkpoint_file):
    """Plot and save the training/validation loss curves as a PNG file."""
    if stat_iter and (stat_loss_train or stat_loss_val):
        logger.debug(f"Plotting loss curves with {len(stat_iter)} data points")
        logger.debug(f"stat_iter: {stat_iter}")
        logger.debug(f"stat_loss_train: {stat_loss_train}")
        logger.debug(f"stat_loss_val: {stat_loss_val}")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create a figure with two subplots - one for loss, one for learning rate
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot loss curves on the top subplot
            if stat_loss_train:
                ax1.plot(stat_iter, stat_loss_train, label='Train Loss', color='blue', marker='.', alpha=0.7)
            if stat_loss_val:
                ax1.plot(stat_iter, stat_loss_val, label='Val Loss', color='orange', marker='.', alpha=0.7)
            
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training and Validation Loss ({os.path.basename(checkpoint_file)})')
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Add min values as annotations
            if stat_loss_train:
                min_train_idx = stat_loss_train.index(min(stat_loss_train))
                min_train_iter = stat_iter[min_train_idx]
                min_train_loss = stat_loss_train[min_train_idx]
                ax1.annotate(f'Min: {min_train_loss:.4f}', 
                           (min_train_iter, min_train_loss),
                           xytext=(10, -20),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
            
            if stat_loss_val:
                min_val_idx = stat_loss_val.index(min(stat_loss_val))
                min_val_iter = stat_iter[min_val_idx]
                min_val_loss = stat_loss_val[min_val_idx]
                ax1.annotate(f'Min: {min_val_loss:.4f}', 
                           (min_val_iter, min_val_loss),
                           xytext=(10, 20),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
            
            # Plot learning rate on the bottom subplot if we have data
            global stat_lr_batch, stat_lr
            if stat_lr_batch and stat_lr:
                ax2.plot(stat_lr_batch, stat_lr, color='green', marker='.', alpha=0.7)
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.grid(True, linestyle='--', alpha=0.5)
                
                # Use scientific notation for small learning rates
                ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
            # Add timestamp to the plot
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(0.5, 0.01, f"Generated: {timestamp}", ha="center", fontsize=8, 
                       bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout()
            png_path = checkpoint_file + '.png'
            plt.savefig(png_path)
            print(f"{BOLD}{GREEN}Saved loss curve plot to:{ENDC} {png_path}")
            plt.close(fig)
        except ImportError:
            print(f"{YELLOW}Matplotlib not available. Cannot plot loss curves.{ENDC}")
        except Exception as e:
            print(f"{YELLOW}Error plotting loss curves: {e}{ENDC}")
    else:
        print(f"{YELLOW}No loss statistics to plot.{ENDC}")

# TRAINING LOOP
try:
    # Debug info about the data
    logger.debug("========== DATA INFO ==========")
    logger.debug(f"train_data type: {type(train_data)}")
    logger.debug(f"val_data type: {type(val_data)}")
    logger.debug(f"batch_size: {batch_size}, block_size: {block_size}, device: {device}")
    logger.debug(f"max_epochs: {max_epochs}, gradient_accumulation_steps: {gradient_accumulation_steps}")
    logger.debug(f"checkpoint_file: {checkpoint_file}")
    if train_data is None or val_data is None:
        logger.error("Data not properly loaded!")
        sys.exit(1)
        
    t0 = time.time()
    # Two counters are used for different purposes:
    # 1. batch_counter - Counts total batches across all epochs (saved in checkpoints)
    #    Used for tracking statistics and plotting loss/learning rate curves
    #    Already initialized from checkpoint if available, otherwise start at 0
    if 'batch_counter' not in locals():
        batch_counter = 0  
        logger.debug(f"Initialized batch_counter to 0")
    
    # 2. epoch_iter - Tracks the current batch index within the current epoch
    #    Resets to 0 at the start of each epoch and IS saved in checkpoints for resuming training
    #    Used for progress display, learning rate scheduling, and checkpoint resumption
    
    raw_model = model
    running_mfu = -1.0  # Memory Footprint Utilization on GPU
    
    # Configure training using epoch-based approach
    # Create data loader for training
    train_loader = DataLoader(train_data, batch_size, block_size, shuffle=True)
    
    # Set up validation loader if needed
    val_loader = DataLoader(val_data, batch_size, block_size, shuffle=False)
    
    if 'epoch' not in globals():
        # Initialize counters for epoch tracking if not already set
        epoch = 0
        epoch_iter = 0
    
    epoch_loss = 0.0
    train_loader.epoch_start_time = time.time()
    
    # Calculate header length to ensure proper alignment
    header_length = 42  # Fixed width for the header box
    epoch_text = f"EPOCH {epoch+1} OF {max_epochs}"
    padding = " " * ((header_length - len(epoch_text)) // 2)  # Calculate padding for centering
    right_padding = " " * (header_length - len(epoch_text) - len(padding))  # Adjust right padding for exact width
    
    # Add a colorful header for the first epoch
    print(f"\n{BOLD}Starting training: {max_epochs} epochs, {len(train_loader)} batches per epoch{ENDC}\n")

    print(f"{BOLD}{BLUE}╔══════════════════════════════════════════╗{ENDC}")
    print(f"{BOLD}{BLUE}║{padding}{epoch_text}{right_padding}║{ENDC}")
    print(f"{BOLD}{BLUE}╚══════════════════════════════════════════╝{ENDC}\n")
    
    # Log the start of the training session
    log_training_start(args)
    
    # Main training loop
    while True:
        # Break condition for epoch-based training
        if epoch >= max_epochs:
            print(f"{GREEN}Reached maximum number of epochs ({max_epochs}){ENDC}")
            break
        
        try:
            # Get the next batch
            if epoch_iter >= len(train_loader):
                # End of epoch - first print newline to clear the progress line
                print()  # Move to next line for epoch completion output
                
                epoch_duration = time.time() - train_loader.epoch_start_time
                
                # Show 100% completion bar for satisfying closure
                epoch_bar = '█' * 30  # Completely filled bar
                print(f"{YELLOW}[{epoch_bar}]{ENDC} {BOLD}Epoch {epoch+1}/{max_epochs}{ENDC} | "
                      f"{CYAN}Batch {len(train_loader)}/{len(train_loader)}{ENDC} | "
                      f"{MAGENTA}100.0%{ENDC} | "
                      f"Loss: {lossf:.4f} | "
                      f"{GREEN}Complete!{ENDC}")
                
                print(f"\n{BOLD}{GREEN}==== Epoch {epoch+1}/{max_epochs} Complete ===={ENDC}")
                print(f"{CYAN}Duration: {str(datetime.timedelta(seconds=int(epoch_duration)))}{ENDC}")
                print(f"{CYAN}Average Loss: {epoch_loss/epoch_iter:.4f}{ENDC}")
                
                # Run evaluation at the end of each epoch
                losses = estimate_loss()
                print(f"{GREEN}Train Loss: {losses['train']:.4f}{ENDC}  {MAGENTA}Val Loss: {losses['val']:.4f}{ENDC}")
                print(f"{YELLOW}Progress: {epoch+1}/{max_epochs} epochs ({(epoch+1)/max_epochs*100:.1f}%){ENDC}\n")
                
                # After evaluation at the end of each epoch, before save_model and plot_loss_curves:
                if stat_iter is not None:
                    # Record stats using batch counter instead of epoch number
                    stat_iter.append(batch_counter)
                    stat_loss_train.append(losses['train'])
                    stat_loss_val.append(losses['val'])
                    logger.debug(f"Recording end-of-epoch stats at batch {batch_counter}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
                
                # Save checkpoint at the end of each epoch
                save_success = save_model(checkpoint_file)
                plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, checkpoint_file)
                if not save_success:
                    logger.warning(f"{YELLOW}Failed to save checkpoint after epoch {epoch+1}. Will try again later.{ENDC}")
                
                # Prepare for next epoch
                epoch += 1
                if epoch >= max_epochs:
                    # We're done with training
                    break
                    
                # Reset for the next epoch
                epoch_iter = 0
                epoch_loss = 0.0
                train_loader.epoch_start_time = time.time()
                
                # Calculate header length to ensure proper alignment
                header_length = 42  # Fixed width for the header box
                epoch_text = f"EPOCH {epoch+1} OF {max_epochs}"
                padding = " " * ((header_length - len(epoch_text)) // 2)  # Calculate padding for centering
                right_padding = " " * (header_length - len(epoch_text) - len(padding))  # Adjust right padding for exact width
                
                # Add a colorful header for the new epoch
                print(f"\n{BOLD}{BLUE}╔══════════════════════════════════════════╗{ENDC}")
                print(f"{BOLD}{BLUE}║{padding}{epoch_text}{right_padding}║{ENDC}")
                print(f"{BOLD}{BLUE}╚══════════════════════════════════════════╝{ENDC}\n")
                
                # Get batch from train_loader using iterator protocol
                if epoch_iter == 0:
                    # Create a new iterator at the start of each epoch
                    train_iter = iter(train_loader)
                
                try:
                    X, Y = next(train_iter)
                    # Move data to device
                    if device_type == 'cuda':
                        X = X.pin_memory().to(device, non_blocking=True)
                        Y = Y.pin_memory().to(device, non_blocking=True)
                    else:
                        X = X.to(device)
                        Y = Y.to(device)
                    epoch_iter += 1
                except StopIteration:
                    # This shouldn't happen if we reset properly at the end of each epoch
                    # but just in case
                    logger.warning(f"{YELLOW}StopIteration encountered mid-epoch. This shouldn't happen.{ENDC}")
                    continue
            else:
                # Get batch from training dataset
                X, Y = get_batch('train')
                # Need to increment epoch_iter here too when using get_batch() directly
                epoch_iter += 1
            
            # FORWARD PROP, update with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                
                # Removed iteration-based prefetching logic that was previously here
                    
                # with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                
            # clip the gradient - to avoid exploding gradients
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # Apply learning rate schedule based on epoch progress
            if decay_lr:
                # Calculate current position in training (as a fraction of an epoch)
                epoch_fraction = epoch + (epoch_iter / len(train_loader))
                lr = get_lr(epoch_fraction)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Record learning rate for plotting
                if epoch_iter % max(1, len(train_loader) // 10) == 0:
                    stat_lr_batch.append(batch_counter)
                    stat_lr.append(lr)
                    logger.debug(f"Recording LR at batch {batch_counter}: {lr}")
            
            # BACK PROP
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
        
            # stats for timing
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            # Track epoch loss for reporting
            epoch_loss += lossf
            
            if batch_counter >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # Real-time progress update for every batch using \r (carriage return)
            # Calculate progress percentage for epoch
            epoch_progress = epoch_iter / len(train_loader) * 100
            bar_length = 30
            epoch_bar_filled = int(bar_length * epoch_iter / len(train_loader))
            epoch_bar = '█' * epoch_bar_filled + '░' * (bar_length - epoch_bar_filled)
            
            # Calculate ETA and other metrics
            if epoch_iter > 0:
                epoch_start_time = getattr(train_loader, 'epoch_start_time', time.time())
                time_per_batch = (time.time() - epoch_start_time) / epoch_iter
                eta_seconds = time_per_batch * (len(train_loader) - epoch_iter)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                samples_per_sec = batch_size / dt if dt > 0 else 0
                
                # Get current learning rate if available
                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else learning_rate
                
                # Real-time progress display (overwrites same line with \r)
                progress_line = (f"{YELLOW}[{epoch_bar}]{ENDC} {BOLD}Epoch {epoch+1}/{max_epochs}{ENDC} | "
                               f"{CYAN}Batch {epoch_iter}/{len(train_loader)}{ENDC} | "
                               f"{MAGENTA}{epoch_progress:.1f}%{ENDC} | "
                               f"Loss: {lossf:.4f} | "
                               f"LR: {current_lr:.2e} | "
                               f"{GREEN}ETA: {eta_str}{ENDC} | "
                               f"Samples/s: {samples_per_sec:.1f}")
                
                if running_mfu > 0:
                    progress_line += f" | MFU: {running_mfu*100:.1f}%"
                
                # Print progress line with \r to overwrite previous line
                print(progress_line, flush=True)
            else:
                # Set start time when beginning a new epoch
                if not hasattr(train_loader, 'epoch_start_time'):
                    train_loader.epoch_start_time = time.time()
                
                # Initial progress display
                progress_line = (f"{YELLOW}[{epoch_bar}]{ENDC} {BOLD}Epoch {epoch+1}/{max_epochs}{ENDC} | "
                               f"{CYAN}Batch {epoch_iter}/{len(train_loader)}{ENDC} | "
                               f"{MAGENTA}{epoch_progress:.1f}%{ENDC} | "
                               f"Loss: {lossf:.4f} | Starting...")
                print(progress_line, flush=True)
            
            # Periodic evaluation during an epoch (based on percentage completed)
            eval_percentage = args.eval_interval
            if eval_percentage > 0 and eval_percentage < 100:
                epoch_percentage = (epoch_iter / len(train_loader)) * 100
                if int(epoch_percentage) % eval_percentage == 0 and epoch_percentage > 0:
                    # Print newline before evaluation to preserve progress line
                    print()  # Move to next line for evaluation output
                    
                    # Only evaluate once at each percentage point
                    if not hasattr(train_loader, 'last_eval_point') or int(epoch_percentage) > train_loader.last_eval_point:
                        train_loader.last_eval_point = int(epoch_percentage)
                        
                        # Run mid-epoch evaluation
                        losses = estimate_loss()
                        print(f"\n{GREEN}Epoch {epoch+1} ({epoch_percentage:.1f}%): Train Loss: {losses['train']:.4f}{ENDC}  {MAGENTA}Val Loss: {losses['val']:.4f}{ENDC}")
                        
                        # Record stats at this intermediate checkpoint
                        if stat_iter is not None:
                            stat_iter.append(batch_counter)
                            stat_loss_train.append(losses['train'])
                            stat_loss_val.append(losses['val'])
                            logger.debug(f"Recording mid-epoch stats at batch {batch_counter}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
                        
                        # Save intermediate checkpoint if requested
                        intermediate_checkpoint = checkpoint_file.replace('.pt', f'_e{epoch+1}_p{int(epoch_percentage)}.pt')
                        print(f"{YELLOW}Saving intermediate checkpoint...{ENDC}")
                        save_success = save_model(intermediate_checkpoint)
                        
                        # Generate and save the loss curve plot
                        plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, intermediate_checkpoint)
                        
                        if save_success:
                            print(f"{GREEN}Intermediate checkpoint saved successfully.{ENDC}")
                        else:
                            print(f"{RED}Failed to save intermediate checkpoint.{ENDC}")
            
            batch_counter += 1
            logger.debug(f"Processed batch {batch_counter}, epoch {epoch}, iter {epoch_iter}/{len(train_loader)}")
        
            # Check if we've reached the end of the epoch
            if epoch_iter >= len(train_loader):
                break
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.info("Saving checkpoint due to exception.")
            save_success = save_model(checkpoint_file)
            plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, checkpoint_file)
            if not save_success:
                logger.error(f"{BOLD}{RED}Failed to save checkpoint after exception. Data may be lost.{ENDC}")
                
            # Log the failed training session
            training_duration = time.time() - t0
            latest_train_loss = stat_loss_train[-1] if stat_loss_train else None
            latest_val_loss = stat_loss_val[-1] if stat_loss_val else None
            best_val_loss = min(stat_loss_val) if stat_loss_val else None
            
            log_training_end(
                success=False,
                duration=training_duration,
                train_loss=latest_train_loss,
                val_loss=latest_val_loss,
                best_val_loss=best_val_loss,
                exception=f"{type(e).__name__}: {str(e)}"
            )
            
            raise
except KeyboardInterrupt:
    print(f"\n{BOLD}{RED}User interrupted training at epoch {epoch+1}, batch {epoch_iter}/{len(train_loader)}.{ENDC}")
    logger.info("Running final evaluation and saving checkpoint before exit...")
    
    # Run a final evaluation to get accurate loss values
    try:
        final_losses = estimate_loss()
        final_train_loss = final_losses['train']
        final_val_loss = final_losses['val']
        
        # Record final stats
        if stat_iter:  # Only append if we have existing stats
            stat_iter.append(batch_counter)  # Use batch counter for x-axis
            stat_loss_train.append(final_train_loss)
            stat_loss_val.append(final_val_loss)
            logger.debug(f"Recording intermediate checkpoint stats at batch {batch_counter}: train_loss={final_train_loss:.4f}, val_loss={final_val_loss:.4f}")
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}")
        final_train_loss = stat_loss_train[-1] if stat_loss_train else "N/A"
        final_val_loss = stat_loss_val[-1] if stat_loss_val else "N/A"
    
    # Save the checkpoint
    save_success = save_model(checkpoint_file)
    plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, checkpoint_file)
    if save_success:
        print(f"{BOLD}{GREEN}Checkpoint saved successfully.{ENDC}")
    else:
        print(f"\n{BOLD}{RED}Failed to save checkpoint! Your progress may be lost.{ENDC}")
    
    # Log interrupted training session
    training_duration = time.time() - t0
    best_val_loss = min(stat_loss_val) if stat_loss_val else None
    
    # Convert non-float values to None for logging
    train_loss = final_train_loss if isinstance(final_train_loss, float) else None
    val_loss = final_val_loss if isinstance(final_val_loss, float) else None
    
    log_training_end(
        success=False,
        duration=training_duration,
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        exception="KeyboardInterrupt: Training manually interrupted by user"
    )
    
    # Display training summary
    print(f"\n{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
    print(f"{BOLD}{BLUE}║                TRAINING INTERRUPTED                   ║{ENDC}")
    print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
    print(f"{BOLD}Completed:{ENDC} {GREEN}{epoch} epochs and {epoch_iter}/{len(train_loader)} batches{ENDC}")
    
    # Format losses properly based on their type
    if isinstance(final_train_loss, float):
        train_loss_str = f"{GREEN}{final_train_loss:.4f}{ENDC}"
    else:
        train_loss_str = f"{GREEN}{final_train_loss}{ENDC}"
        
    if isinstance(final_val_loss, float):
        val_loss_str = f"{GREEN}{final_val_loss:.4f}{ENDC}"
    else:
        val_loss_str = f"{GREEN}{final_val_loss}{ENDC}"
    
    print(f"{BOLD}Final train loss:{ENDC}     {train_loss_str}")
    print(f"{BOLD}Final val loss:{ENDC}       {val_loss_str}")
    print(f"{BOLD}Checkpoint saved to:{ENDC} {GREEN}{checkpoint_file}{ENDC}")
    
# Final save and summary at normal completion
if epoch >= max_epochs:
    # Display final 100% progress bar for overall training
    total_bar = '█' * 30  # Completely filled bar
    print(f"\n{YELLOW}[{total_bar}]{ENDC} {BOLD}Training complete!{ENDC} | "
          f"{MAGENTA}100.0%{ENDC} | "
          f"All {max_epochs} epochs finished")
    
    print(f"\n{BOLD}{BLUE}Training completed. Running final evaluation...{ENDC}")
    
    # Run a final evaluation to get accurate loss values
    final_losses = estimate_loss()
    
    # Record final stats
    if stat_iter:  # Only append if we have existing stats
        stat_iter.append(batch_counter)  # Use batch counter for x-axis
        stat_loss_train.append(final_losses['train'])
        stat_loss_val.append(final_losses['val'])
        logger.debug(f"Recording final stats at batch {batch_counter}: train_loss={final_losses['train']:.4f}, val_loss={final_losses['val']:.4f}")
    
    # Save the model with updated stats
    save_success = save_model(checkpoint_file)
    plot_loss_curves(stat_iter, stat_loss_train, stat_loss_val, checkpoint_file)
    if save_success:
        print(f"{BOLD}{GREEN}Checkpoint saved successfully.{ENDC}")
    else:
        print(f"\n{BOLD}{RED}Failed to save final checkpoint! Your training results may be lost.{ENDC}")
    
    # Get best validation loss
    best_val_loss = min(stat_loss_val) if stat_loss_val else final_losses['val']
    
    # Log successful training completion
    training_duration = time.time() - t0
    log_training_end(
        success=True,
        duration=training_duration,
        train_loss=final_losses['train'],
        val_loss=final_losses['val'],
        best_val_loss=best_val_loss
    )
    
    print(f"\n{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
    print(f"{BOLD}{BLUE}║                TRAINING COMPLETED                     ║{ENDC}")
    print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
    print(f"{BOLD}Total training time:{ENDC}  {GREEN}{str(datetime.timedelta(seconds=int(time.time() - t0)))}{ENDC}")
    print(f"{BOLD}Final train loss:{ENDC}     {GREEN}{final_losses['train']:.4f}{ENDC}")
    print(f"{BOLD}Final val loss:{ENDC}       {GREEN}{final_losses['val']:.4f}{ENDC}")
    print(f"{BOLD}Best val loss:{ENDC}        {GREEN}{best_val_loss:.4f}{ENDC}")
    print(f"{BOLD}Model saved to:{ENDC}       {GREEN}{checkpoint_file}{ENDC}")
    print(f"\n{BOLD}{CYAN}To generate text with this model, run:{ENDC}")
    print(f"{YELLOW}python gen.py {checkpoint_file} --chat{ENDC}" if "chat" in dataset_name else f"{YELLOW}python gen.py {checkpoint_file}{ENDC}")
    print(f"\n{BOLD}{GREEN}Training successfully completed!{ENDC}")