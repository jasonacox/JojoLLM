#!/usr/bin/python3
"""
Jojo LLM Training Script

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
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

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

# Define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model on various datasets')
    parser.add_argument('--dataset', type=str, default='story', 
                        help='Dataset name to use for training (default: story)')
    parser.add_argument('--max_iters', type=int, default=100, 
                        help='Total number of training iterations (default: 5000)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for (overrides max_iters if specified)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility (default: 1337)')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='How often to run evaluation (default: 500)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training progress (default: 10)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to continue training from')
    parser.add_argument('--output_checkpoint', type=str, default=None,
                        help='Custom path to save the output checkpoint (defaults to models/{dataset}{max_iters}.pt)')
    parser.add_argument('--reset_iter', action='store_true',
                        help='Reset iteration counter to 0 when loading from checkpoint')
    parser.add_argument('--epoch_mode', action='store_true',
                        help='Use epoch-based training instead of iteration-based training')
    return parser.parse_args()

args = parse_args()

# Disable colors if requested
if hasattr(args, 'no_color') and args.no_color:
    GREEN = YELLOW = BLUE = RED = MAGENTA = CYAN = ENDC = BOLD = UNDERLINE = ""

# Set up logging
logging.basicConfig(level=logging.INFO, format=f'{BLUE}%(asctime)s{ENDC} {GREEN}%(levelname)s:{ENDC} %(message)s')
logger = logging.getLogger(__name__)

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
eval_iters = 200 # number of iterations to run evaluation
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
lr_decay_iters = 60000 # [600000] should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda:1' # examples: 'cpu', 'cuda', or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16'

###############################################################################

# We already defined parse_args() earlier, no need to redefine it

# Use the args object we already created earlier
# Set dataset-specific parameters
dataset_name = args.dataset
print(f"Using dataset: {dataset_name}")

# Set the target max iterations
max_iters = args.max_iters

# Set checkpoint file based on dataset and iterations or use custom path if specified
if args.output_checkpoint:
    checkpoint_file = args.output_checkpoint
else:
    checkpoint_file = f'models/{dataset_name}{max_iters}.pt'

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
iter_num = 0

# capture above settings & parameters to save in model checkpoint
config_keys = [k for k,v in globals().items() 
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))
               and not k in ('args', 'parser')]
config = {k: globals()[k] for k in config_keys} 

# Tokens per iterations
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size

# Load the data
data_dir = 'data/'
try:
    train_data = np.memmap(os.path.join(data_dir, f'{dataset_name}-train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, f'{dataset_name}-val.bin'), dtype=np.uint16, mode='r')
    print(f"Loaded {dataset_name} dataset: {len(train_data):,} training tokens, {len(val_data):,} validation tokens")
except FileNotFoundError:
    print(f"Error: Could not find {dataset_name} dataset files.")
    print(f"Make sure to run data/prepare-{dataset_name}.py first.")
    sys.exit(1)

# Calculate total tokens based on training mode
if args.epoch_mode:
    # For epoch-based training, estimate tokens based on dataset size and epochs
    data_size = len(train_data)
    tokens_per_epoch = data_size - (data_size % (batch_size * block_size))  # Approximate tokens seen per epoch
    if args.epochs is not None:
        max_epochs = args.epochs
    else:
        # Calculate epochs from max_iters (approximate)
        batches_per_epoch = tokens_per_epoch // (batch_size * block_size)
        max_epochs = max_iters // batches_per_epoch if batches_per_epoch > 0 else 1
    total_tokens = tokens_per_epoch * max_epochs
else:
    # For iteration-based training
    total_tokens = tokens_per_iter * max_iters  # We're starting from iteration 0

print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
print(f"{BOLD}{BLUE}║                TRAINING CONFIGURATION                 ║{ENDC}")
print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
print(f"{BOLD}Dataset:{ENDC}          {GREEN}{dataset_name}{ENDC}")
print(f"{BOLD}Training mode:{ENDC}    {GREEN}{'Epoch-based' if args.epoch_mode else 'Iteration-based'}{ENDC}")

if args.epoch_mode:
    # Show epoch information
    if 'epoch' in globals() and epoch > 0:
        print(f"{BOLD}Starting epoch:{ENDC}   {GREEN}{epoch+1}/{max_epochs}{ENDC}")
    else:
        print(f"{BOLD}Epochs to run:{ENDC}    {GREEN}{max_epochs}{ENDC}")
    print(f"{BOLD}Batches per epoch:{ENDC}{GREEN}{len(train_data) // (batch_size * block_size)}{ENDC}")
else:
    # Show iteration information
    if iter_num > 0:
        print(f"{BOLD}Starting iteration:{ENDC} {GREEN}{iter_num:,}{ENDC}")
    print(f"{BOLD}Max iterations:{ENDC}   {GREEN}{max_iters:,}{ENDC}")
    print(f"{BOLD}Iterations to run:{ENDC} {GREEN}{max_iters - iter_num:,}{ENDC}")
print(f"{BOLD}Batch size:{ENDC}       {GREEN}{batch_size}{ENDC}")
print(f"{BOLD}Block size:{ENDC}       {GREEN}{block_size}{ENDC}")
print(f"{BOLD}Learning rate:{ENDC}    {GREEN}{learning_rate}{ENDC}")
print(f"{BOLD}Tokens per iter:{ENDC}  {GREEN}{tokens_per_iter:,}{ENDC}")
print(f"{BOLD}Remaining tokens:{ENDC} {GREEN}{total_tokens:,}{ENDC}")
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

# create an epoch-based data loader
class DataLoader:
    def __init__(self, data, batch_size, block_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.n_batches = (len(data) - block_size) // block_size // batch_size
        if self.n_batches == 0:
            raise ValueError(f"Dataset too small ({len(data)} tokens) for batch size {batch_size} and block size {block_size}")
        self.batch_idx = 0  # Initialize batch_idx in constructor
        self.epoch_start_time = time.time()  # Track when the epoch started
        self._create_batches()

    def _create_batches(self):
        # Create indices for all possible starting positions
        self.indices = list(range(len(self.data) - self.block_size))
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Group indices into batches
        self.batches = []
        for i in range(0, min(len(self.indices), self.n_batches * self.batch_size), self.batch_size):
            if i + self.batch_size <= len(self.indices):
                self.batches.append(self.indices[i:i+self.batch_size])

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= len(self.batches):
            raise StopIteration
        
        # Get current batch indices
        batch_indices = self.batches[self.batch_idx]
        self.batch_idx += 1
        
        # Create x and y tensors
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in batch_indices])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in batch_indices])
        
        return x, y
        
    def __len__(self):
        return len(self.batches)

# get a batch from the data (for backward compatibility with the evaluation function)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
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
        
    # If we want to continue counting iterations from checkpoint
    if 'iter_num' in checkpoint:
        checkpoint_iter = checkpoint['iter_num']
        
        # Load epoch data if available
        checkpoint_epoch = checkpoint.get('epoch', 0)
        checkpoint_epoch_iter = checkpoint.get('epoch_iter', 0)
        
        if not args.reset_iter:
            # Get target iterations based on mode
            target_iters = max_iters
            if args.epoch_mode and args.epochs is not None:
                print(f"{GREEN}Continuing epoch-based training from epoch {checkpoint_epoch+1}{ENDC}")
            
            # Check if the model is already trained beyond our target max_iters or epochs
            if checkpoint_iter >= max_iters:
                print(f"\n{YELLOW}Warning: The loaded model has already been trained for {checkpoint_iter} iterations,{ENDC}")
                print(f"{YELLOW}which is more than or equal to your requested max_iters ({max_iters}).{ENDC}")
                if args.epoch_mode and 'epoch' in checkpoint:
                    print(f"{YELLOW}The model completed {checkpoint_epoch+1} epochs already.{ENDC}")
                print(f"{YELLOW}This means no additional training will occur unless you increase iterations/epochs or use --reset_iter.{ENDC}\n")
                
                while True:
                    response = input(f"{BOLD}How do you want to proceed?{ENDC}\n"
                                    f"1) Continue with current settings (no additional training)\n"
                                    f"2) Reset iteration/epoch counters to 0\n"
                                    f"3) Increase {'epochs' if args.epoch_mode else 'max_iters'}\n"
                                    f"4) Quit\n"
                                    f"Enter choice [1-4]: ").strip()
                    
                    if response == '1':
                        print(f"{YELLOW}Continuing with current settings (iter_num={checkpoint_iter}){ENDC}")
                        iter_num = checkpoint_iter
                        break
                    elif response == '2':
                        print(f"{YELLOW}Resetting iteration counter from {checkpoint_iter} to 0{ENDC}")
                        iter_num = 0
                        args.reset_iter = True  # Update flag for consistency
                        break
                    elif response == '3':
                        while True:
                            try:
                                new_max = int(input(f"Enter new max_iters value (greater than {checkpoint_iter}): "))
                                if new_max > checkpoint_iter:
                                    max_iters = new_max
                                    print(f"{GREEN}Updated max_iters to {max_iters}{ENDC}")
                                    iter_num = checkpoint_iter
                                    # Update checkpoint file path if it's based on max_iters
                                    if not args.output_checkpoint:
                                        checkpoint_file = f'models/{dataset_name}{max_iters}.pt'
                                        print(f"{GREEN}Updated output checkpoint path to: {checkpoint_file}{ENDC}")
                                    break
                                else:
                                    print(f"{RED}Error: New max_iters must be greater than {checkpoint_iter}{ENDC}")
                            except ValueError:
                                print(f"{RED}Please enter a valid integer{ENDC}")
                        break
                    elif response == '4':
                        print(f"{YELLOW}Exiting at user request{ENDC}")
                        sys.exit(0)
                    else:
                        print(f"{RED}Invalid choice. Please enter a number between 1 and 4.{ENDC}")
            else:
                # Normal case - continue from checkpoint iteration
                iter_num = checkpoint_iter
                print(f"{GREEN}Continuing from iteration {iter_num}{ENDC}")
        else:
            # Reset iterations as requested
            print(f"{YELLOW}Resetting iteration counter from {checkpoint_iter} to 0{ENDC}")
            iter_num = 0
        print(f"Continuing from iteration {iter_num}")
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
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad() # disable gradient tracking - not needed for backward pass
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Save model to checkpoint file
# Add resume option
resume = False
if os.path.exists(checkpoint_file):
    logger.info(f"Checkpoint {checkpoint_file} found. Use resume option to continue training from checkpoint.")
    # Optionally, load checkpoint here if resume is True

def save_model(fn):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    
    # Create checkpoint data
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config,
    }
    # Add dataset information to the checkpoint
    checkpoint['dataset'] = dataset_name
    # Add epoch information if using epoch-based training
    if args.epoch_mode and 'epoch' in globals():
        checkpoint['epoch'] = epoch
        checkpoint['epoch_iter'] = epoch_iter
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

# stats
stat_iter = []
stat_loss_train = []
stat_loss_val = []
stat_lr_iter = []
stat_lr = []

# TRAINING LOOP
try:
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0  # Memory Footprint Utilization on GPU
    
    # Configure the training mode (epoch-based or iteration-based)
    if args.epoch_mode:
        # Create data loader for epoch-based training
        train_loader = DataLoader(train_data, batch_size, block_size, shuffle=True)
        
        # Set up validation loader if needed
        val_loader = None  # Uncomment to create a validation loader
        
        # Initialize counters for epoch tracking
        epoch = 0
        epoch_iter = 0
        epoch_loss = 0.0
        train_loader.epoch_start_time = time.time()
        
        # Determine number of epochs to run
        max_epochs = args.epochs if args.epochs is not None else 1
        
        # Calculate header length to ensure proper alignment
        header_length = 42  # Fixed width for the header box
        epoch_text = f"EPOCH 1 OF {max_epochs}"
        padding = " " * ((header_length - len(epoch_text)) // 2)  # Calculate padding for centering
        right_padding = " " * (header_length - len(epoch_text) - len(padding))  # Adjust right padding for exact width
        
        # Add a colorful header for the first epoch
        print(f"\n{BOLD}Starting training in epoch mode: {max_epochs} epochs, {len(train_loader)} batches per epoch{ENDC}\n")
    
        print(f"{BOLD}{BLUE}╔══════════════════════════════════════════╗{ENDC}")
        print(f"{BOLD}{BLUE}║{padding}{epoch_text}{right_padding}║{ENDC}")
        print(f"{BOLD}{BLUE}╚══════════════════════════════════════════╝{ENDC}\n")     

        # Show total iterations that will be performed
        max_iters = max_epochs * len(train_loader)
    else:
        # Traditional iteration-based training
        # Fetch the very first batch
        X, Y = get_batch('train')
        print(f"\n{BOLD}Starting training in iteration mode: {max_iters} iterations{ENDC}\n")
    
    # Main training loop
    while True:
        # Break condition for iteration-based training
        if not args.epoch_mode and iter_num >= max_iters:
            # Show 100% progress bar
            bar = '█' * 30  # Completely filled bar
            print(f"{YELLOW}[{bar}]{ENDC} {BOLD}{max_iters}/{max_iters}{ENDC} | " 
                  f"{CYAN}100.0%{ENDC} | "
                  f"{GREEN}Training complete!{ENDC}")
            
            print(f"{GREEN}Reached maximum number of iterations ({max_iters}){ENDC}")
            break
            
        # Break condition for epoch-based training
        if args.epoch_mode and epoch >= max_epochs:
            print(f"{GREEN}Reached maximum number of epochs ({max_epochs}){ENDC}")
            break
        
        try:
            # Get the next batch based on training mode
            if args.epoch_mode:
                # For epoch-based training
                if epoch_iter >= len(train_loader):
                    # End of epoch
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
                    
                    # Save checkpoint at the end of each epoch
                    save_success = save_model(checkpoint_file)
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
                    train_loader._create_batches()  # Re-shuffle the data
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
                # For iteration-based training, use the get_batch function
                X, Y = get_batch('train')
            
            # FORWARD PROP, update with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                
                # For iteration-based training, immediately async prefetch next batch while model is doing the forward pass
                if not args.epoch_mode:
                    X, Y = get_batch('train')
                    
                # with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                
            # clip the gradient - to avoid exploding gradients
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            if args.epoch_mode:
                epoch_loss += lossf
            
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # Create a progress bar for every 10 iterations
            if iter_num % 10 == 0:
                # Calculate progress percentage
                progress = iter_num / max_iters * 100 if max_iters > 0 else 0
                bar_length = 30
                filled_length = int(bar_length * iter_num / max_iters) if max_iters > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                if args.epoch_mode:
                    # Enhanced progress indication for epoch-based training
                    epoch_progress = epoch_iter / len(train_loader) * 100
                    epoch_bar_filled = int(bar_length * epoch_iter / len(train_loader))
                    epoch_bar = '█' * epoch_bar_filled + '░' * (bar_length - epoch_bar_filled)
                    
                    # Calculate ETA for current epoch
                    if epoch_iter > 0:
                        epoch_start_time = getattr(train_loader, 'epoch_start_time', t0)
                        time_per_batch = (time.time() - epoch_start_time) / epoch_iter
                        eta_seconds = time_per_batch * (len(train_loader) - epoch_iter)
                        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                        samples_per_sec = batch_size / dt if dt > 0 else 0
                        
                        print(f"{YELLOW}[{epoch_bar}]{ENDC} {BOLD}Epoch {epoch+1}/{max_epochs}{ENDC} | "
                              f"{CYAN}Batch {epoch_iter}/{len(train_loader)}{ENDC} | "
                              f"{MAGENTA}{epoch_progress:.1f}%{ENDC} | "
                              f"Loss: {lossf:.4f} | "
                              f"{GREEN}ETA: {eta_str}{ENDC} | "
                              f"Samples/s: {samples_per_sec:.1f}")
                    else:
                        # Set start time when beginning a new epoch
                        if not hasattr(train_loader, 'epoch_start_time'):
                            train_loader.epoch_start_time = time.time()
                        print(f"{YELLOW}[{epoch_bar}]{ENDC} {BOLD}Epoch {epoch+1}/{max_epochs}{ENDC} | "
                              f"{CYAN}Batch {epoch_iter}/{len(train_loader)}{ENDC} | "
                              f"{MAGENTA}{epoch_progress:.1f}%{ENDC} | "
                              f"Loss: {lossf:.4f}")
                else:
                    print(f"{YELLOW}[{bar}]{ENDC} {BOLD}{iter_num}/{max_iters}{ENDC} | " 
                          f"{CYAN}Loss: {lossf:.4f}{ENDC} | "
                          f"{GREEN}Time: {dt*1000:.2f}ms{ENDC} | "
                          f"{MAGENTA}MFU: {running_mfu*100:.2f}%{ENDC}")
            
            iter_num += 1
            local_iter_num += 1
        
            # termination conditions
            if args.epoch_mode:
                if epoch >= max_epochs:
                    break
            else:
                if iter_num >= max_iters:
                    break
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.info("Saving checkpoint due to exception.")
            save_success = save_model(checkpoint_file)
            if not save_success:
                logger.error(f"{BOLD}{RED}Failed to save checkpoint after exception. Data may be lost.{ENDC}")
            raise
except KeyboardInterrupt:
    print(f"\n{BOLD}{RED}User interrupted training at iteration {iter_num}.{ENDC}")
    logger.info("Running final evaluation and saving checkpoint before exit...")
    
    # Run a final evaluation to get accurate loss values
    try:
        final_losses = estimate_loss()
        final_train_loss = final_losses['train']
        final_val_loss = final_losses['val']
        
        # Record final stats
        if stat_iter:  # Only append if we have existing stats
            stat_iter.append(iter_num)
            stat_loss_train.append(final_train_loss)
            stat_loss_val.append(final_val_loss)
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}")
        final_train_loss = stat_loss_train[-1] if stat_loss_train else "N/A"
        final_val_loss = stat_loss_val[-1] if stat_loss_val else "N/A"
    
    # Save the checkpoint
    save_success = save_model(checkpoint_file)
    if not save_success:
        print(f"\n{BOLD}{RED}Failed to save checkpoint! Your progress may be lost.{ENDC}")
    
    # Display training summary
    print(f"\n{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
    print(f"{BOLD}{BLUE}║                TRAINING INTERRUPTED                   ║{ENDC}")
    print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
    print(f"{BOLD}Completed iterations:{ENDC} {GREEN}{iter_num}/{max_iters}{ENDC}")
    
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
if iter_num >= max_iters:
    # Display final 100% progress bar for overall training
    if args.epoch_mode:
        total_bar = '█' * 30  # Completely filled bar
        print(f"\n{YELLOW}[{total_bar}]{ENDC} {BOLD}Training complete!{ENDC} | "
              f"{MAGENTA}100.0%{ENDC} | "
              f"All {max_epochs} epochs finished")
    
    print(f"\n{BOLD}{BLUE}Training completed. Running final evaluation...{ENDC}")
    
    # Run a final evaluation to get accurate loss values
    final_losses = estimate_loss()
    
    # Record final stats
    if stat_iter:  # Only append if we have existing stats
        stat_iter.append(iter_num)
        stat_loss_train.append(final_losses['train'])
        stat_loss_val.append(final_losses['val'])
    
    # Save the model with updated stats
    save_success = save_model(checkpoint_file)
    if not save_success:
        print(f"\n{BOLD}{RED}Failed to save final checkpoint! Your training results may be lost.{ENDC}")
    
    # Get best validation loss
    best_val_loss = min(stat_loss_val) if stat_loss_val else final_losses['val']
    
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