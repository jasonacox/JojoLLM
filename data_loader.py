#!/usr/bin/python3
"""
Improved data loading for Jojo LLM Training

This module provides optimized data loading classes with pre-tokenization,
caching, and efficient batch generation.

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import os
import json
import pickle
import random
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from config import Constants
from utils import TensorBuffer


logger = logging.getLogger(__name__)


class CachedJsonlDataset(Dataset):
    """JSONL dataset with pre-tokenization and caching"""
    
    def __init__(self, jsonl_file: str, tokenizer: Any, cache_dir: str = "cache", 
                 block_size: int = 1024, force_retokenize: bool = False):
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = cache_dir
        
        # Handle caching only if cache_dir is provided
        if cache_dir:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache file path
            cache_filename = f"{os.path.basename(jsonl_file)}_bs{block_size}.cache"
            self.cache_file = os.path.join(cache_dir, cache_filename)
            
            # Load or create cached data
            if os.path.exists(self.cache_file) and not force_retokenize:
                logger.info(f"Loading cached tokenized data from {self.cache_file}")
                self._load_cached_data()
            else:
                logger.info(f"Tokenizing and caching data from {jsonl_file}")
                self._tokenize_and_cache()
        else:
            # No caching - tokenize directly
            logger.info(f"Tokenizing data from {jsonl_file} (no caching)")
            self.cache_file = None
            self._tokenize_and_cache(save_cache=False)
        
        # Initialize epoch tracking
        self.reset_epoch()
        
        logger.info(f"Dataset loaded: {len(self.conversations)} conversations, "
                   f"{self.total_tokens:,} total tokens")
    
    def _load_cached_data(self) -> None:
        """Load pre-tokenized data from cache"""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.conversations = cache_data['conversations']
        self.total_tokens = cache_data['total_tokens']
        self.special_token_stats = cache_data.get('special_token_stats', {})
        self.metadata = cache_data.get('metadata', {})
    
    def _tokenize_and_cache(self, save_cache: bool = True) -> None:
        """Tokenize conversations and cache the results"""
        self.conversations = []
        self.total_tokens = 0
        self.special_token_stats = {}
        
        # Load and tokenize conversations
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conversation = json.loads(line.strip())
                    text = conversation.get("text", "")
                    
                    if not text:
                        logger.warning(f"Empty text in line {line_num}, skipping")
                        continue
                    
                    # Tokenize the conversation
                    tokens = self._tokenize_text(text)
                    
                    if len(tokens) == 0:
                        logger.warning(f"Empty tokens for line {line_num}, skipping")
                        continue
                    
                    self.conversations.append({
                        'tokens': tokens,
                        'length': len(tokens),
                        'line_num': line_num
                    })
                    self.total_tokens += len(tokens)
                    
                    # Update special token statistics
                    self._update_special_token_stats(tokens)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
        
        # Create metadata
        self.metadata = {
            'original_file': self.jsonl_file,
            'block_size': self.block_size,
            'total_conversations': len(self.conversations),
            'total_tokens': self.total_tokens,
            'tokenizer_type': str(type(self.tokenizer))
        }
        
        # Cache the tokenized data only if save_cache is True and cache_file is set
        if save_cache and self.cache_file:
            cache_data = {
                'conversations': self.conversations,
                'total_tokens': self.total_tokens,
                'special_token_stats': self.special_token_stats,
                'metadata': self.metadata
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cached tokenized data to {self.cache_file}")
        elif not save_cache:
            logger.info("Tokenization completed (caching disabled)")
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text with fallback handling"""
        try:
            # Try to tokenize with special tokens allowed
            return self.tokenizer.encode(text, allowed_special="all")
        except Exception:
            try:
                # Fallback to standard encoding
                return self.tokenizer.encode(text)
            except Exception as e:
                logger.error(f"Tokenization failed: {e}")
                return []
    
    def _update_special_token_stats(self, tokens: List[int]) -> None:
        """Update special token statistics"""
        # This is a simplified version - you might want to add
        # specific logic for counting special tokens
        for token in tokens:
            if token >= 50257:  # Assuming special tokens are > vocab_size
                token_name = f"special_{token}"
                self.special_token_stats[token_name] = self.special_token_stats.get(token_name, 0) + 1
    
    def reset_epoch(self) -> None:
        """Reset for a new epoch"""
        self.used_indices = set()
        logger.debug("Reset epoch - cleared used indices")
    
    def __len__(self) -> int:
        """Return number of conversations"""
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item (not typically used in batch generation)"""
        conversation = self.conversations[idx]
        tokens = conversation['tokens']
        
        # Create input/target pairs
        if len(tokens) > self.block_size:
            # Take a random slice if conversation is longer than block_size
            max_start = len(tokens) - self.block_size - 1
            start_idx = random.randint(0, max_start)
            x = torch.tensor(tokens[start_idx:start_idx + self.block_size], dtype=torch.long)
            y = torch.tensor(tokens[start_idx + 1:start_idx + self.block_size + 1], dtype=torch.long)
        else:
            # Pad if necessary
            x_tokens = tokens[:-1] if len(tokens) > 1 else tokens
            y_tokens = tokens[1:] if len(tokens) > 1 else tokens
            
            x = torch.tensor(x_tokens, dtype=torch.long)
            y = torch.tensor(y_tokens, dtype=torch.long)
            
            # Pad to block_size
            if len(x) < self.block_size:
                padding_x = torch.zeros(self.block_size - len(x), dtype=torch.long)
                padding_y = torch.zeros(self.block_size - len(y), dtype=torch.long)
                x = torch.cat([x, padding_x])
                y = torch.cat([y, padding_y])
        
        return {'input': x, 'target': y}
    
    def get_batch_data(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get a batch of data ensuring proper epoch coverage.
        Each conversation is used exactly once per epoch before reuse.
        """
        # Use tensor buffer for efficiency
        if not hasattr(self, '_tensor_buffer') or self._tensor_buffer is None:
            buffer_device = device or torch.device('cpu')
            self._tensor_buffer = TensorBuffer(batch_size, self.block_size, buffer_device)
        
        x_batch, y_batch = self._tensor_buffer.get_buffers()
        epoch_complete = False
        
        for b in range(batch_size):
            # Check if we need to reset for a new epoch
            if len(self.used_indices) >= len(self.conversations):
                logger.debug(f"Completed full epoch through {len(self.conversations)} conversations")
                self.reset_epoch()
                epoch_complete = True
            
            # Get available conversations
            available_indices = [i for i in range(len(self.conversations)) if i not in self.used_indices]
            
            if not available_indices:
                logger.warning("No available conversations - resetting epoch")
                self.reset_epoch()
                available_indices = list(range(len(self.conversations)))
                epoch_complete = True
            
            # Choose a random unused conversation
            conversation_idx = random.choice(available_indices)
            self.used_indices.add(conversation_idx)
            
            # Get the conversation tokens
            tokens = self.conversations[conversation_idx]['tokens']
            
            if len(tokens) == 0:
                logger.warning(f"Empty conversation at index {conversation_idx}")
                x_batch[b] = torch.zeros(self.block_size, dtype=torch.long)
                y_batch[b] = torch.zeros(self.block_size, dtype=torch.long)
                continue
            
            # Process tokens into input/target pairs
            if len(tokens) > self.block_size + 1:
                # Take a random slice
                max_start = len(tokens) - self.block_size - 1
                start_idx = random.randint(0, max_start)
                x = tokens[start_idx:start_idx + self.block_size]
                y = tokens[start_idx + 1:start_idx + self.block_size + 1]
            else:
                # Use available tokens and pad if necessary
                x = tokens[:-1] if len(tokens) > 1 else tokens
                y = tokens[1:] if len(tokens) > 1 else tokens
                
                # Pad to block_size
                if len(x) < self.block_size:
                    x = x + [0] * (self.block_size - len(x))
                if len(y) < self.block_size:
                    y = y + [0] * (self.block_size - len(y))
            
            # Convert to tensors and add to batch
            x_batch[b] = torch.tensor(x[:self.block_size], dtype=torch.long)
            y_batch[b] = torch.tensor(y[:self.block_size], dtype=torch.long)
        
        # Move to device if specified and not already there
        if device is not None and x_batch.device != device:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        
        return x_batch, y_batch, epoch_complete
    
    def print_dataset_stats(self) -> None:
        """Print dataset statistics"""
        print(f"{Constants.BOLD}Dataset Statistics:{Constants.ENDC}")
        print(f"  - Total conversations: {len(self.conversations):,}")
        print(f"  - Total tokens: {self.total_tokens:,}")
        print(f"  - Average tokens per conversation: {self.total_tokens / len(self.conversations):.1f}")
        print(f"  - Block size: {self.block_size}")
        
        if self.special_token_stats:
            print(f"  - Special token counts:")
            for token, count in self.special_token_stats.items():
                print(f"    - {token}: {count:,}")
        print()


class EfficientDataLoader:
    """Efficient data loader with proper epoch handling"""
    
    def __init__(self, dataset: CachedJsonlDataset, batch_size: int, 
                 device: Optional[torch.device] = None, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
        # Calculate batches per epoch (ceiling division)
        self.batches_per_epoch = max(1, (len(dataset) + batch_size - 1) // batch_size)
        
        self.reset_epoch()
        
        logger.info(f"DataLoader initialized: {self.batches_per_epoch} batches per epoch "
                   f"for {len(dataset)} conversations")
    
    def reset_epoch(self) -> None:
        """Reset for new epoch"""
        self.current_batch = 0
        self.epoch_start_time = time.time()
        self.dataset.reset_epoch()
    
    def __iter__(self):
        """Iterator protocol"""
        self.reset_epoch()
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch"""
        if self.current_batch >= self.batches_per_epoch:
            raise StopIteration
        
        x, y, epoch_complete = self.dataset.get_batch_data(self.batch_size, self.device)
        self.current_batch += 1
        
        # If epoch completed early, we can stop iteration
        if epoch_complete and self.current_batch >= len(self.dataset) // self.batch_size:
            raise StopIteration
        
        return x, y
    
    def __len__(self) -> int:
        """Return number of batches per epoch"""
        return self.batches_per_epoch


def create_dataloaders(train_file: str, val_file: str, tokenizer: Any, 
                      batch_size: int, block_size: int, cache_dir: str = "cache",
                      device: Optional[torch.device] = None, num_workers: int = 0) -> Tuple[EfficientDataLoader, EfficientDataLoader]:
    """Create train and validation data loaders"""
    
    logger.info("Creating data loaders...")
    
    # Create datasets
    train_dataset = CachedJsonlDataset(train_file, tokenizer, cache_dir, block_size)
    val_dataset = CachedJsonlDataset(val_file, tokenizer, cache_dir, block_size)
    
    # Print dataset statistics
    print(f"{Constants.GREEN}Training Data:{Constants.ENDC}")
    train_dataset.print_dataset_stats()
    
    print(f"{Constants.GREEN}Validation Data:{Constants.ENDC}")
    val_dataset.print_dataset_stats()
    
    # Create data loaders
    train_loader = EfficientDataLoader(train_dataset, batch_size, device, shuffle=True)
    val_loader = EfficientDataLoader(val_dataset, batch_size, device, shuffle=False)
    
    return train_loader, val_loader


def get_batch_legacy(dataset: CachedJsonlDataset, batch_size: int, block_size: int, 
                    device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy batch function for backward compatibility"""
    x, y, _ = dataset.get_batch_data(batch_size, device)
    return x, y
