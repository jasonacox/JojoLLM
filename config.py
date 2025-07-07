#!/usr/bin/python3
"""
Configuration management for Jojo LLM Training

This module provides configuration classes and utilities for managing
training parameters, model settings, and system configurations.

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Union


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.2
    bias: bool = False
    vocab_size: int = 50304


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 60000
    min_lr: float = 6e-5
    warmup_fraction: float = 0.1
    cooldown_fraction: float = 0.9


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    max_epochs: int = 1
    max_iters: int = None  # Maximum iterations (overrides epochs if set)
    batch_size: int = 12                    # Number of sequences per batch
    gradient_accumulation_steps: int = 40   # Effective batch = batch_size * gradient_accumulation_steps
    eval_iters: int = 200
    eval_interval: int = 50
    log_interval: int = 50
    save_checkpoints: bool = True
    checkpoint_interval: int = 0  # Save checkpoint every N batches (0 = only at epoch end)
    compile_model: bool = True
    # Packed data loader settings - None means use entire dataset per epoch
    train_batches: int = None  # Number of batches per epoch for training (None = entire dataset)
    val_batches: int = None    # Number of batches per epoch for validation (None = entire dataset)


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    seed: int = 1337
    num_workers: int = 4
    pin_memory: bool = True
    # Memory optimization settings
    memory_fraction: float = 0.9  # CUDA memory fraction to use
    optimize_memory: bool = True  # Enable memory optimizations
    # TF32 optimization settings for modern GPUs
    allow_tf32_matmul: bool = True  # Enable TF32 for matrix multiplications
    allow_tf32_cudnn: bool = True   # Enable TF32 for cuDNN operations


@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    dataset_name: str = 'chitchat'
    data_dir: str = 'data/'
    cache_tokenized: bool = True
    cache_dir: str = 'cache/'


@dataclass
class Config:
    """Master configuration class"""
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    system: SystemConfig
    data: DataConfig
    
    def __init__(self, **kwargs):
        self.model = ModelConfig(**kwargs.get('model', {}))
        self.optimizer = OptimizerConfig(**kwargs.get('optimizer', {}))
        self.scheduler = SchedulerConfig(**kwargs.get('scheduler', {}))
        self.training = TrainingConfig(**kwargs.get('training', {}))
        self.system = SystemConfig(**kwargs.get('system', {}))
        self.data = DataConfig(**kwargs.get('data', {}))
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments"""
        if hasattr(args, 'dataset') and args.dataset is not None:
            self.data.dataset_name = args.dataset
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.training.max_epochs = args.epochs
        if hasattr(args, 'max_iters') and args.max_iters is not None:
            self.training.max_iters = args.max_iters
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            self.optimizer.learning_rate = args.learning_rate
        if hasattr(args, 'eval_interval') and args.eval_interval is not None:
            self.training.eval_interval = args.eval_interval
        if hasattr(args, 'log_interval') and args.log_interval is not None:
            self.training.log_interval = args.log_interval
        if hasattr(args, 'checkpoint_interval') and args.checkpoint_interval is not None:
            self.training.checkpoint_interval = args.checkpoint_interval
        if hasattr(args, 'train_batches') and args.train_batches is not None:
            self.training.train_batches = args.train_batches
        if hasattr(args, 'val_batches') and args.val_batches is not None:
            self.training.val_batches = args.val_batches
        if hasattr(args, 'seed') and args.seed is not None:
            self.system.seed = args.seed


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def validate_config(config: Config) -> None:
    """Validate configuration parameters"""
    assert config.training.batch_size > 0, "Batch size must be positive"
    assert config.optimizer.learning_rate > 0, "Learning rate must be positive"
    assert config.training.max_epochs > 0, "Number of epochs must be positive"
    assert config.model.n_layer > 0, "Number of layers must be positive"
    assert config.model.n_head > 0, "Number of heads must be positive"
    assert config.model.n_embd > 0, "Embedding dimension must be positive"
    assert config.model.block_size > 0, "Block size must be positive"
    assert config.training.gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
    
    # Check if device is available
    if config.system.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA device specified but CUDA is not available")
    
    # Check data directory exists
    if not os.path.exists(config.data.data_dir):
        raise ValueError(f"Data directory {config.data.data_dir} does not exist")


# Constants
class Constants:
    # Version information
    VERSION = "2.1.0"
    VERSION_NAME = "Complete Training System"
    
    DEFAULT_VOCAB_SIZE = 50304
    PROGRESS_BAR_LENGTH = 30
    MFU_WARMUP_BATCHES = 5
    CHECKPOINT_TEMP_SUFFIX = ".tmp"
    LOG_FILE = "train.log"
    
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    UNDERLINE = "\033[4m"
