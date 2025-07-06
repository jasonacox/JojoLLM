#!/usr/bin/python3
"""
Jojo LLM Training Script - Refactored Version

This is the improved version of the training script with better organization,
efficiency, and maintainability.

Version: 2.0.0 "Modular Architecture"

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import os
import sys
import argparse
import logging
import torch

# Import our refactored modules
from config import Config, get_default_config, validate_config, Constants
from utils import Logger, DeviceManager
from trainer import Trainer
from model import GPTConfig, GPT


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a GPT model on various datasets')
    
    # Dataset and training
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to use for training (default: use config file value)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for (default: use config file value)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: use config file value)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: use config file value)')
    
    # Evaluation and logging
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='How often to evaluate during training (in batches, default: use config file value)')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='How often to log progress (in batches, default: use config file value)')
    
    # System
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: use config file value)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Data type for training (default: use config file value)')
    
    # Checkpoints
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to continue training from')
    parser.add_argument('--output_checkpoint', type=str, default=None,
                        help='Custom path to save output checkpoint')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--save_config', type=str, default=None,
                        help='Save current configuration to file')
    parser.add_argument('--show_config', action='store_true',
                        help='Show complete configuration and exit')
    
    # Other options
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable tokenization caching')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable model compilation')
    parser.add_argument('--no_color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--version', action='store_true',
                        help='Show version information and exit')
    
    return parser.parse_args()


def show_version():
    """Show version information"""
    print(f"Jojo LLM Training Script")
    print(f"Version: {Constants.VERSION} \"{Constants.VERSION_NAME}\"")
    print(f"PyTorch: {torch.__version__}")
    print(f"Author: Jason A. Cox")
    print(f"https://github.com/jasonacox/jojo")
    
    # Show git commit if available
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            commit = result.stdout.strip()[:8]
            print(f"Git commit: {commit}")
    except Exception:
        pass


def setup_environment(args):
    """Setup environment and configuration"""
    
    # Disable colors if requested
    if args.no_color:
        for attr in ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'BOLD', 'ENDC', 'UNDERLINE']:
            setattr(Constants, attr, "")
    
    # Setup logging
    Logger.setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Print header
    print(f"{Constants.BOLD}{Constants.BLUE}╔═══════════════════════════════════════════════════════╗{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}║                Jojo LLM Training Program              ║{Constants.ENDC}")
    
    # Center version info
    version_info = f"Version {Constants.VERSION} \"{Constants.VERSION_NAME}\""
    box_width = 55  # Internal width of the box (57 chars - 2 border chars)
    padding_left = (box_width - len(version_info)) // 2
    padding_right = box_width - len(version_info) - padding_left  # Calculate exact right padding
    
    print(f"{Constants.BOLD}{Constants.BLUE}║{' ' * padding_left}{version_info}{' ' * padding_right}║{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}╚═══════════════════════════════════════════════════════╝{Constants.ENDC}")
    print()
    
    return logger


def create_configuration(args):
    """Create and validate configuration"""
    
    # Load configuration from file if specified
    if args.config:
        if os.path.exists(args.config):
            config = Config.from_file(args.config)
            print(f"{Constants.GREEN}Loaded configuration from {args.config}{Constants.ENDC}")
        else:
            print(f"{Constants.RED}Configuration file {args.config} not found{Constants.ENDC}")
            sys.exit(1)
    else:
        # Use default configuration
        config = get_default_config()
    
    # Update configuration from command line arguments
    config.update_from_args(args)
    
    # Override specific settings from args only if provided
    if args.device:
        config.system.device = args.device
    else:
        # Auto-detect best device
        config.system.device = DeviceManager.select_best_device()
    
    if args.dtype is not None:
        config.system.dtype = args.dtype
    if args.seed is not None:
        config.system.seed = args.seed
    if args.no_cache:
        config.data.cache_tokenized = False
    if args.no_compile:
        config.training.compile_model = False
    
    # Validate configuration
    validate_config(config)
    
    # Save configuration if requested
    if args.save_config:
        config.to_file(args.save_config)
        print(f"{Constants.GREEN}Configuration saved to {args.save_config}{Constants.ENDC}")
    
    return config


def setup_model(config):
    """Initialize model"""
    logger = logging.getLogger(__name__)
    
    # Create model configuration
    model_config = GPTConfig(
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        block_size=config.model.block_size,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout
    )
    
    # Initialize model
    model = GPT(model_config)
    
    # Move to device
    device = torch.device(config.system.device)
    model.to(device)
    
    # Model compilation
    if config.training.compile_model and not config.system.device == 'mps':
        logger.info("Compiling model (this may take a minute)...")
        model = torch.compile(model)
    elif config.system.device == 'mps':
        logger.info("MPS doesn't support JIT compilation, skipping...")
    
    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{Constants.BOLD}{Constants.BLUE}╔═══════════════════════════════════════════════════════╗{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}║             MODEL ARCHITECTURE SUMMARY                ║{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}╚═══════════════════════════════════════════════════════╝{Constants.ENDC}")
    print(f"{Constants.BOLD}Total parameters:{Constants.ENDC} {Constants.GREEN}{param_count/1e6:.2f}M{Constants.ENDC}")
    print(f"{Constants.BOLD}Model configuration:{Constants.ENDC}")
    print(f"  - Layers: {config.model.n_layer}")
    print(f"  - Heads: {config.model.n_head}")
    print(f"  - Embedding dim: {config.model.n_embd}")
    print(f"  - Block size: {config.model.block_size}")
    print(f"  - Dropout: {config.model.dropout}")
    print(f"  - Bias: {config.model.bias}")
    print()
    
    return model


def setup_tokenizer():
    """Initialize tokenizer"""
    try:
        # Try to use extended tokenizer with special token support
        if os.path.exists('setup_tokenizer.py'):
            from setup_tokenizer import get_extended_tokenizer
            tokenizer = get_extended_tokenizer()
            print(f"{Constants.GREEN}Using extended tokenizer with special token support{Constants.ENDC}")
            return tokenizer
        else:
            raise ImportError("setup_tokenizer.py not found")
    except Exception as e:
        print(f"{Constants.YELLOW}Extended tokenizer not available ({e}){Constants.ENDC}")
        print(f"{Constants.YELLOW}Falling back to standard tiktoken...{Constants.ENDC}")
        
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"{Constants.YELLOW}Using cl100k_base tokenizer (limited special token support){Constants.ENDC}")
            return tokenizer
        except Exception as e2:
            print(f"{Constants.RED}Failed to initialize tokenizer: {e2}{Constants.ENDC}")
            sys.exit(1)


def print_training_config(config):
    """Print training configuration summary"""
    print(f"{Constants.BOLD}{Constants.BLUE}╔═══════════════════════════════════════════════════════╗{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}║                TRAINING CONFIGURATION                 ║{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}╚═══════════════════════════════════════════════════════╝{Constants.ENDC}")
    print(f"{Constants.BOLD}Dataset:{Constants.ENDC}          {Constants.GREEN}{config.data.dataset_name}{Constants.ENDC}")
    print(f"{Constants.BOLD}Epochs:{Constants.ENDC}           {Constants.GREEN}{config.training.max_epochs}{Constants.ENDC}")
    print(f"{Constants.BOLD}Batch size:{Constants.ENDC}       {Constants.GREEN}{config.training.batch_size}{Constants.ENDC}")
    print(f"{Constants.BOLD}Block size:{Constants.ENDC}       {Constants.GREEN}{config.model.block_size}{Constants.ENDC}")
    print(f"{Constants.BOLD}Learning rate:{Constants.ENDC}    {Constants.GREEN}{config.optimizer.learning_rate}{Constants.ENDC}")
    print(f"{Constants.BOLD}Gradient accum:{Constants.ENDC}   {Constants.GREEN}{config.training.gradient_accumulation_steps}{Constants.ENDC}")
    print(f"{Constants.BOLD}Device:{Constants.ENDC}           {Constants.GREEN}{config.system.device}{Constants.ENDC}")
    print(f"{Constants.BOLD}Precision:{Constants.ENDC}        {Constants.GREEN}{config.system.dtype}{Constants.ENDC}")
    print(f"{Constants.BOLD}Model compilation:{Constants.ENDC} {Constants.GREEN}{'Enabled' if config.training.compile_model else 'Disabled'}{Constants.ENDC}")
    print(f"{Constants.BOLD}Tokenization cache:{Constants.ENDC} {Constants.GREEN}{'Enabled' if config.data.cache_tokenized else 'Disabled'}{Constants.ENDC}")
    print()


def show_complete_config(config):
    """Display complete configuration in a formatted way"""
    print(f"{Constants.BOLD}{Constants.BLUE}╔═══════════════════════════════════════════════════════╗{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}║                COMPLETE CONFIGURATION                 ║{Constants.ENDC}")
    print(f"{Constants.BOLD}{Constants.BLUE}╚═══════════════════════════════════════════════════════╝{Constants.ENDC}")
    
    # Model Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}Model Configuration:{Constants.ENDC}")
    print(f"  n_layer:           {Constants.GREEN}{config.model.n_layer}{Constants.ENDC}")
    print(f"  n_head:            {Constants.GREEN}{config.model.n_head}{Constants.ENDC}")
    print(f"  n_embd:            {Constants.GREEN}{config.model.n_embd}{Constants.ENDC}")
    print(f"  block_size:        {Constants.GREEN}{config.model.block_size}{Constants.ENDC}")
    print(f"  dropout:           {Constants.GREEN}{config.model.dropout}{Constants.ENDC}")
    print(f"  bias:              {Constants.GREEN}{config.model.bias}{Constants.ENDC}")
    print(f"  vocab_size:        {Constants.GREEN}{config.model.vocab_size}{Constants.ENDC}")
    print()
    
    # Optimizer Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}Optimizer Configuration:{Constants.ENDC}")
    print(f"  learning_rate:     {Constants.GREEN}{config.optimizer.learning_rate}{Constants.ENDC}")
    print(f"  weight_decay:      {Constants.GREEN}{config.optimizer.weight_decay}{Constants.ENDC}")
    print(f"  beta1:             {Constants.GREEN}{config.optimizer.beta1}{Constants.ENDC}")
    print(f"  beta2:             {Constants.GREEN}{config.optimizer.beta2}{Constants.ENDC}")
    print(f"  grad_clip:         {Constants.GREEN}{config.optimizer.grad_clip}{Constants.ENDC}")
    print()
    
    # Scheduler Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}Scheduler Configuration:{Constants.ENDC}")
    print(f"  decay_lr:          {Constants.GREEN}{config.scheduler.decay_lr}{Constants.ENDC}")
    print(f"  warmup_iters:      {Constants.GREEN}{config.scheduler.warmup_iters}{Constants.ENDC}")
    print(f"  lr_decay_iters:    {Constants.GREEN}{config.scheduler.lr_decay_iters}{Constants.ENDC}")
    print(f"  min_lr:            {Constants.GREEN}{config.scheduler.min_lr}{Constants.ENDC}")
    print(f"  warmup_fraction:   {Constants.GREEN}{config.scheduler.warmup_fraction}{Constants.ENDC}")
    print(f"  cooldown_fraction: {Constants.GREEN}{config.scheduler.cooldown_fraction}{Constants.ENDC}")
    print()
    
    # Training Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}Training Configuration:{Constants.ENDC}")
    print(f"  max_epochs:               {Constants.GREEN}{config.training.max_epochs}{Constants.ENDC}")
    print(f"  batch_size:               {Constants.GREEN}{config.training.batch_size}{Constants.ENDC}")
    print(f"  gradient_accumulation_steps: {Constants.GREEN}{config.training.gradient_accumulation_steps}{Constants.ENDC}")
    print(f"  eval_iters:               {Constants.GREEN}{config.training.eval_iters}{Constants.ENDC}")
    print(f"  eval_interval:            {Constants.GREEN}{config.training.eval_interval}{Constants.ENDC}")
    print(f"  log_interval:             {Constants.GREEN}{config.training.log_interval}{Constants.ENDC}")
    print(f"  save_checkpoints:         {Constants.GREEN}{config.training.save_checkpoints}{Constants.ENDC}")
    print(f"  compile_model:            {Constants.GREEN}{config.training.compile_model}{Constants.ENDC}")
    print()
    
    # System Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}System Configuration:{Constants.ENDC}")
    print(f"  device:            {Constants.GREEN}{config.system.device}{Constants.ENDC}")
    print(f"  dtype:             {Constants.GREEN}{config.system.dtype}{Constants.ENDC}")
    print(f"  seed:              {Constants.GREEN}{config.system.seed}{Constants.ENDC}")
    print(f"  num_workers:       {Constants.GREEN}{config.system.num_workers}{Constants.ENDC}")
    print(f"  pin_memory:        {Constants.GREEN}{config.system.pin_memory}{Constants.ENDC}")
    print()
    
    # Data Configuration
    print(f"{Constants.BOLD}{Constants.CYAN}Data Configuration:{Constants.ENDC}")
    print(f"  dataset_name:      {Constants.GREEN}{config.data.dataset_name}{Constants.ENDC}")
    print(f"  data_dir:          {Constants.GREEN}{config.data.data_dir}{Constants.ENDC}")
    print(f"  cache_tokenized:   {Constants.GREEN}{config.data.cache_tokenized}{Constants.ENDC}")
    print(f"  cache_dir:         {Constants.GREEN}{config.data.cache_dir}{Constants.ENDC}")
    print()


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle version request
    if args.version:
        show_version()
        sys.exit(0)
    
    # Setup environment
    logger = setup_environment(args)
    logger.info("--- Jojo LLM Training Script Started ---")
    
    try:
        # Create configuration
        config = create_configuration(args)
        
        # Handle show config request
        if args.show_config:
            show_complete_config(config)
            sys.exit(0)
        
        # Print configuration
        print_training_config(config)
        
        # Set random seed
        import random
        import numpy as np
        random.seed(config.system.seed)
        np.random.seed(config.system.seed)
        torch.manual_seed(config.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.system.seed)
        
        # Setup tokenizer
        tokenizer = setup_tokenizer()
        
        # Setup model
        model = setup_model(config)
        
        # Optimize device memory
        DeviceManager.optimize_memory(config.system.device)
        
        # Create trainer
        trainer = Trainer(config, model, tokenizer)
        
        # Determine checkpoint paths
        checkpoint_path = args.output_checkpoint
        if not checkpoint_path:
            checkpoint_path = f"models/{config.data.dataset_name}_epoch{config.training.max_epochs}.pt"
        
        resume_from = args.checkpoint if args.resume else None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train(checkpoint_path=checkpoint_path, resume_from=resume_from)
        
        if results['success']:
            # Training completed successfully
            print(f"\n{Constants.BOLD}{Constants.BLUE}╔═══════════════════════════════════════════════════════╗{Constants.ENDC}")
            print(f"{Constants.BOLD}{Constants.BLUE}║                TRAINING COMPLETED                     ║{Constants.ENDC}")
            print(f"{Constants.BOLD}{Constants.BLUE}╚═══════════════════════════════════════════════════════╝{Constants.ENDC}")
            print(f"{Constants.BOLD}Total training time:{Constants.ENDC}  {Constants.GREEN}{results['total_time']:.0f}s{Constants.ENDC}")
            print(f"{Constants.BOLD}Final train loss:{Constants.ENDC}     {Constants.GREEN}{results['final_train_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Final val loss:{Constants.ENDC}       {Constants.GREEN}{results['final_val_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Best train loss:{Constants.ENDC}      {Constants.GREEN}{results['best_train_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Best val loss:{Constants.ENDC}        {Constants.GREEN}{results['best_val_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Worst train loss:{Constants.ENDC}     {Constants.RED}{results['worst_train_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Worst val loss:{Constants.ENDC}       {Constants.RED}{results['worst_val_loss']:.4f}{Constants.ENDC}")
            print(f"{Constants.BOLD}Model saved to:{Constants.ENDC}       {Constants.GREEN}{checkpoint_path}{Constants.ENDC}")
            print(f"\n{Constants.BOLD}{Constants.CYAN}To generate text with this model, run:{Constants.ENDC}")
            if "chat" in config.data.dataset_name:
                print(f"{Constants.YELLOW}python gen.py {checkpoint_path} --chat{Constants.ENDC}")
            else:
                print(f"{Constants.YELLOW}python gen.py {checkpoint_path}{Constants.ENDC}")
            print(f"\n{Constants.BOLD}{Constants.GREEN}Training successfully completed!{Constants.ENDC}")
            
        else:
            # Training failed
            print(f"\n{Constants.BOLD}{Constants.RED}Training failed: {results.get('error', 'Unknown error')}{Constants.ENDC}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Constants.BOLD}{Constants.YELLOW}Training interrupted by user{Constants.ENDC}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        print(f"\n{Constants.BOLD}{Constants.RED}Training failed: {e}{Constants.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
