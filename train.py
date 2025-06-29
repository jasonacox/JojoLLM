max_iters = 5000 # total number of training iterations
checkpoint_file = f'models/story{max_iters}.pt'

import os
import time
import math
import random
from contextlib import nullcontext
import logging

import numpy as np
import torch
import numpy as np
from model import GPTConfig, GPT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
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

# capture above settings & parameters to save in model checkpoint
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} 

# tokens per iterations
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

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
data_dir = dataset + '/'
train_data = np.memmap(os.path.join(data_dir, 'story-train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'story-val.bin'), dtype=np.uint16, mode='r')

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

# init a new model from scratch
print("Initializing a new model from scratch")
print("Using vocab_size of GPT-2 of 50304 (50257 rounded up for efficiency)")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout) 
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# report number of parameters
print("Number of parameters: %.2fM" % (model.get_num_params()/1e6,))
print("Layers:")
for number, (name, param) in enumerate(model.named_parameters()):
    print(number, name)

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
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config,
    }
    logger.info(f"Saving checkpoint to {fn}")
    torch.save(checkpoint, fn)

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
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if iter_num > 0:
                    save_model(checkpoint_file)
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
            if iter_num % 10 == 0:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1
        
            # termination conditions
            if iter_num > max_iters:
                break
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.info("Saving checkpoint due to exception.")
            save_model(checkpoint_file)
            raise
except KeyboardInterrupt:
    logger.info("User requested exit, saving checkpoint.")
    save_model(checkpoint_file)

save_model(checkpoint_file)