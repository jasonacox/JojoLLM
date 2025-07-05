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
    batch_size: int = 12
    gradient_accumulation_steps: int = 40
    eval_iters: int = 200
    eval_interval: int = 50
    log_interval: int = 50
    save_checkpoints: bool = True
    compile_model: bool = True


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    seed: int = 1337
    num_workers: int = 4
    pin_memory: bool = True


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
        if hasattr(args, 'dataset'):
            self.data.dataset_name = args.dataset
        if hasattr(args, 'epochs'):
            self.training.max_epochs = args.epochs
        if hasattr(args, 'batch_size'):
            self.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate'):
            self.optimizer.learning_rate = args.learning_rate
        if hasattr(args, 'eval_interval'):
            self.training.eval_interval = args.eval_interval
        if hasattr(args, 'log_interval'):
            self.training.log_interval = args.log_interval
        if hasattr(args, 'seed'):
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
    VERSION = "2.0.0"
    VERSION_NAME = "Modular Architecture"
    
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
