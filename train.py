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
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# Define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model on various datasets')
    parser.add_argument('--dataset', type=str, default='story', choices=['story', 'dailydialog', 'chat', 'chitchat'],
                        help='Dataset to use for training (default: story)')
    parser.add_argument('--max_iters', type=int, default=100, 
                        help='Total number of training iterations (default: 5000)')
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
    return parser.parse_args()

args = parse_args()

# Disable colors if requested
if hasattr(args, 'no_color') and args.no_color:
    GREEN = YELLOW = BLUE = RED = MAGENTA = CYAN = ENDC = BOLD = UNDERLINE = ""

# Set up logging
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
if args.dataset == 'story':
    dataset_name = 'story'
elif args.dataset == 'dailydialog':
    dataset_name = 'dailydialog'
elif args.dataset == 'chat':
    dataset_name = 'chat'
    print("Using Human-Assistant chat template dataset.")
elif args.dataset == 'chitchat':
    dataset_name = 'chitchat'
    print("Using diverse chitchat dataset with name personalization.")
else:
    print(f"Unknown dataset: {args.dataset}")
    sys.exit(1)

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

# capture above settings & parameters to save in model checkpoint
config_keys = [k for k,v in globals().items() 
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))
               and not k in ('args', 'parser')]
config = {k: globals()[k] for k in config_keys} 

# tokens per iterations
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
total_tokens = tokens_per_iter * (max_iters - iter_num)  # Account for already completed iterations

print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{ENDC}")
print(f"{BOLD}{BLUE}║                TRAINING CONFIGURATION                 ║{ENDC}")
print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{ENDC}")
print(f"{BOLD}Dataset:{ENDC}          {GREEN}{dataset_name}{ENDC}")
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
    # CUDA device selection
    selected_device = None
    print("CUDA devices available:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # Free memory in GB
        mem_total = props.total_memory / (1024 ** 3)  # Total memory in GB
        print(f"  [{i}] {props.name} - Free: {mem_free:.2f} GB / Total: {mem_total:.2f} GB")
    while True:
        try:
            user_input = input(f"Select CUDA device [0-{torch.cuda.device_count()-1}]: ")
            selected_device = int(user_input)
            if 0 <= selected_device < torch.cuda.device_count():
                break
            else:
                print("Invalid device index.")
        except Exception:
            print("Please enter a valid integer.")
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

# get a batch from the data
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

# loop counters starting point
iter_num = 0

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
        
        if not args.reset_iter:
            # Check if the model is already trained beyond our target max_iters
            if checkpoint_iter >= max_iters:
                print(f"\n{YELLOW}Warning: The loaded model has already been trained for {checkpoint_iter} iterations,{ENDC}")
                print(f"{YELLOW}which is more than or equal to your requested max_iters ({max_iters}).{ENDC}")
                print(f"{YELLOW}This means no additional training will occur unless you increase max_iters or use --reset_iter.{ENDC}\n")
                
                while True:
                    response = input(f"{BOLD}How do you want to proceed?{ENDC}\n"
                                    f"1) Continue with current settings (no additional training)\n"
                                    f"2) Reset iteration counter to 0\n"
                                    f"3) Increase max_iters\n"
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
    # init loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0 # Memory Footprint Utilization on GPU

    while True:
        try:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            if lr > 0:
                stat_lr.append(lr)
                stat_lr_iter.append(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % 100 == 0 or iter_num >= max_iters:
                # Call estimate_loss which handles model.eval() and model.train() internally
                losses = estimate_loss()
                
                # Format nicer output with colors
                elapsed = time.time() - t0
                progress = iter_num / max_iters * 100 if max_iters > 0 else 0
                eta_seconds = (max_iters - iter_num) * (elapsed / (iter_num + 1))
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(f"{BOLD}{BLUE}══════════════════════════════════════════════════════════{ENDC}")
                print(f"{BOLD}{BLUE}STEP {iter_num}/{max_iters}{ENDC} {YELLOW}[{progress:.1f}%]{ENDC} {MAGENTA}ETA: {eta_str}{ENDC}")
                print(f"{GREEN}TRAIN LOSS: {losses['train']:.4f}{ENDC}  {CYAN}VAL LOSS: {losses['val']:.4f}{ENDC}")
                print(f"{BOLD}{BLUE}══════════════════════════════════════════════════════════{ENDC}")
                
                if iter_num > 0:
                    save_success = save_model(checkpoint_file)
                    if not save_success:
                        logger.warning(f"{BOLD}{YELLOW}Failed to save checkpoint at iteration {iter_num}. Will try again later.{ENDC}")
                
                # record stats
                stat_iter.append(iter_num)
                stat_loss_train.append(losses['train'])
                stat_loss_val.append(losses['val'])
        
            # FORWARD PROP, update with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
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
                
                print(f"{YELLOW}[{bar}]{ENDC} {BOLD}{iter_num}/{max_iters}{ENDC} | " 
                      f"{CYAN}Loss: {lossf:.4f}{ENDC} | "
                      f"{GREEN}Time: {dt*1000:.2f}ms{ENDC} | "
                      f"{MAGENTA}MFU: {running_mfu*100:.2f}%{ENDC}")
            
            iter_num += 1
            local_iter_num += 1
        
            # termination conditions
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
    print(f"{BOLD}Final train loss:{ENDC}     {GREEN}{final_train_loss:.4f if isinstance(final_train_loss, float) else final_train_loss}{ENDC}")
    print(f"{BOLD}Final val loss:{ENDC}       {GREEN}{final_val_loss:.4f if isinstance(final_val_loss, float) else final_val_loss}{ENDC}")
    print(f"{BOLD}Checkpoint saved to:{ENDC} {GREEN}{checkpoint_file}{ENDC}")
    
# Final save and summary at normal completion
if iter_num >= max_iters:
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