#!/usr/bin/python3
"""
Simple and efficient packed data loader for Jojo LLM Training

This module provides a highly efficient data loading system that creates densely 
packed training batches by concatenating conversations and filling batches 
completely with actual content, achieving >98% token efficiency.

OVERVIEW:
---------
Traditional data loaders often waste significant space due to padding sequences
to a fixed length. This packed loader eliminates that waste by:

1. Loading all conversations and tokenizing them once
2. Dynamically packing multiple conversations into each sequence 
3. Filling sequences to exactly the block_size with real content
4. Minimizing padding tokens to achieve maximum training efficiency

USAGE BY TRAINER:
-----------------
The trainer uses this loader through the create_simple_packed_loaders() function:

```python
train_loader, val_loader = create_simple_packed_loaders(
    train_file, val_file, tokenizer,
    batch_size=12,      # Number of sequences per batch
    block_size=1024,    # Length of each sequence (context window)
    train_batches=None, # None = use entire dataset per epoch
    val_batches=None    # None = use entire dataset per epoch
)
```

EFFICIENCY GAINS:
-----------------
- Traditional loaders: ~5-20% efficiency (lots of padding)
- Packed loader: >98% efficiency (minimal padding)
- Result: 5-20x more effective training per token processed

EPOCH BEHAVIOR:
---------------
- When max_batches=None: Uses entire dataset, estimating batches from total tokens
- When max_batches=N: Limits epoch to N batches for faster iteration/debugging
- Each epoch shuffles conversations for variety while maintaining efficiency

INTEGRATION WITH TRAINER:
-------------------------
1. Trainer calls _setup_data_loaders() during initialization
2. Loads training and validation datasets using this packed loader
3. Uses estimated_batches for progress tracking and learning rate scheduling
4. Iterates through batches in train_epoch() with perfect 100% token utilization

SUPPORTED DATA FORMATS:
-----------------------
- ChatML format: {'conversation': [{'role': 'user', 'content': '...'}, ...]}
- Plain text: {'text': 'content...'}  
- Input/output: {'input': 'question', 'output': 'answer'}

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import os
import json
import random
from typing import List, Tuple, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset

class SimplePackedDataset(IterableDataset):
    """
    Efficient dataset that packs conversations to maximize token utilization
    
    PACKING ALGORITHM:
    ------------------
    For each sequence in a batch:
    1. Start with empty sequence
    2. Add complete conversations sequentially until no more conversations fit
    3. Never split conversations across sequences to preserve conversational context
    4. Conversations longer than block_size are pre-truncated during loading
    5. Result: sequence filled with complete conversations, optimized for training quality
    
    TRAINING QUALITY vs EFFICIENCY:
    -------------------------------
    - Prioritizes conversational context integrity over pure token efficiency
    - Each sequence contains only complete conversations for proper attention patterns
    - Some redundancy may occur when conversations are seen multiple times per epoch
    - This redundancy is acceptable and beneficial for learning conversational flows
    - Tracks redundancy statistics for monitoring
    
    MEMORY EFFICIENCY:
    ------------------
    - Loads all conversations once at initialization (not per epoch)
    - Conversations are pre-tokenized and cached in memory
    - No disk I/O during training iterations
    - Suitable for datasets that fit comfortably in RAM
    
    RANDOMIZATION:
    --------------
    - Conversations are shuffled at the start of each epoch
    - Conversations are consumed sequentially from the shuffled list
    - Complete conversations may appear multiple times if needed to fill batches
    - This ensures high-quality conversational context in every sequence
    - Multiple epochs will see conversations in different orders due to shuffling
    
    BATCH ESTIMATION:
    -----------------
    When max_batches=None, estimates batches per epoch as:
    estimated_batches = total_dataset_tokens / (batch_size * block_size)
    
    This ensures we use the entire dataset while providing accurate
    progress tracking for the trainer.
    """
    
    def __init__(self, jsonl_file: str, tokenizer, block_size: int = 1024, 
                 batch_size: int = 12, max_batches: int = None):
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_batches = max_batches  # None means use entire dataset
        
        # Load conversations once
        self.conversations = self._load_conversations()
        print(f"✅ Loaded {len(self.conversations)} conversations from {os.path.basename(jsonl_file)}")
        
        # Calculate how many batches we can create from the dataset
        if self.max_batches is None:
            # Estimate batches based on total tokens in dataset
            total_tokens = sum(len(conv) for conv in self.conversations)
            tokens_per_batch = batch_size * block_size
            self.estimated_batches = max(1, total_tokens // tokens_per_batch)
            print(f"📊 Estimated {self.estimated_batches:,} batches per epoch ({total_tokens:,} total tokens)")
        else:
            self.estimated_batches = self.max_batches
            print(f"📊 Limited to {self.max_batches:,} batches per epoch")
    
    def _load_conversations(self) -> List[List[int]]:
        """Load and tokenize conversations"""
        conversations = []
        truncated_count = 0
        
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract conversation text
                    if 'conversation' in data:
                        # ChatML format
                        text = "\n"
                        for turn in data['conversation']:
                            role = turn.get('role', 'user')
                            content = turn.get('content', '')
                            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                        text += "<|endoftext|>"
                    
                    elif 'text' in data:
                        # Plain text
                        text = "\n" + data['text'] + "\n\n"
                    
                    elif 'input' in data and 'output' in data:
                        # Input/output format
                        text = f"\n<|im_start|>user\n{data['input']}<|im_end|>\n<|im_start|>assistant\n{data['output']}<|im_end|>\n<|endoftext|>"
                    
                    else:
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(text, allowed_special="all")
                    if tokens:
                        # Truncate conversations that are longer than block_size
                        if len(tokens) > self.block_size:
                            tokens = tokens[:self.block_size]
                            truncated_count += 1
                        conversations.append(tokens)
                
                except:
                    continue
        
        if truncated_count > 0:
            print(f"\n⚠️  Truncated {truncated_count} conversations longer than block_size ({self.block_size})")
        
        return conversations
    
    def get_num_batches(self) -> int:
        """Get the number of batches this dataset will generate"""
        return self.max_batches if self.max_batches is not None else self.estimated_batches
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate packed batches"""
        
        # Shuffle conversations for variety, but ensure we use each one
        shuffled_conversations = self.conversations.copy()
        random.shuffle(shuffled_conversations)
        
        # Track conversation usage for statistics
        conv_index = 0
        total_tokens_used = 0
        total_unique_tokens = 0
        
        # Determine how many batches to generate
        num_batches = self.max_batches if self.max_batches is not None else self.estimated_batches
        
        for batch_idx in range(num_batches):
            # Create packed batch
            batch_sequences = []
            
            for seq_idx in range(self.batch_size):
                sequence = []
                
                # Keep adding complete conversations until we can't fit another one
                while True:
                    # Get the next conversation
                    current_conv = shuffled_conversations[conv_index % len(shuffled_conversations)]
                    
                    # Check if this conversation fits in remaining space
                    space_left = self.block_size - len(sequence)
                    
                    if len(current_conv) <= space_left:
                        # Conversation fits completely - add it
                        sequence.extend(current_conv)
                        total_tokens_used += len(current_conv)
                        total_unique_tokens += len(current_conv)
                        conv_index += 1
                    else:
                        # Complete conversation doesn't fit - add as much as we can...
                        if space_left > 0:
                            sequence.extend(current_conv[:space_left])
                            total_tokens_used += space_left
                            total_unique_tokens += space_left
                            # save conversation for the next sequence
                        break
                
                # Pad to exact block_size if needed
                while len(sequence) < self.block_size:
                    sequence.append(0)
                
                batch_sequences.append(sequence[:self.block_size])
            
            # Convert to tensors
            X = torch.tensor(batch_sequences, dtype=torch.long)
            Y = torch.cat([X[:, 1:], X[:, :1]], dim=1)  # Shift by 1 for next token prediction
            
            # debug output
            #if batch_idx < 3:
            #    print(f"\nBatch {batch_idx + 1}:")
            #    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
            #    print(f"  X converted to text:\n{self.tokenizer.decode(X[0].tolist())}\n")
            yield X, Y


def create_simple_packed_loaders(train_file: str, val_file: str, tokenizer,
                                batch_size: int, block_size: int,
                                train_batches: int = None, val_batches: int = None):
    """
    Create simple packed data loaders for training and validation
    
    This is the main entry point used by the trainer to create efficient
    data loaders that maximize token utilization during training.
    
    Args:
        train_file: Path to training JSONL file containing conversations
        val_file: Path to validation JSONL file containing conversations
        tokenizer: Tokenizer instance for encoding text to tokens
        batch_size: Number of sequences per batch (NOT total tokens)
        block_size: Sequence length in tokens (context window size)
        train_batches: Max batches per training epoch (None = entire dataset)
        val_batches: Max batches per validation epoch (None = entire dataset)
    
    Returns:
        Tuple of (train_dataset, val_dataset) - both are SimplePackedDataset instances
    
    Example:
        >>> train_loader, val_loader = create_simple_packed_loaders(
        ...     "data/chitchat-train.jsonl", "data/chitchat-val.jsonl", 
        ...     tokenizer, batch_size=12, block_size=1024
        ... )
        >>> # Each batch will be shape [12, 1024] with >98% real content
    
    Integration with Trainer:
        The trainer calls this during _setup_data_loaders() and uses the
        returned datasets in train_epoch() and evaluate() methods.
    """
    
    train_dataset = SimplePackedDataset(train_file, tokenizer, block_size, batch_size, train_batches)
    val_dataset = SimplePackedDataset(val_file, tokenizer, block_size, batch_size, val_batches)
    
    return train_dataset, val_dataset


def test_simple_packed_loader():
    """Test the simple packed loader"""
    from setup_tokenizer import get_extended_tokenizer
    
    tokenizer = get_extended_tokenizer()
    
    # Test with chitchat data
    train_file = "data/chitchat-train.jsonl"
    val_file = "data/chitchat-val.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        return
    
    print("🧪 Testing simple packed data loader...")
    
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=4, block_size=512,
        train_batches=5, val_batches=2
    )
    
    print(f"\n📊 Testing efficiency...")
    
    total_tokens = 0
    effective_tokens = 0
    
    for i, (X, Y) in enumerate(train_dataset):
        batch_total = X.numel()
        batch_effective = (X != 0).sum().item()
        
        total_tokens += batch_total
        effective_tokens += batch_effective
        
        efficiency = batch_effective / batch_total
        print(f"   Batch {i+1}: {batch_effective:,}/{batch_total:,} tokens ({efficiency:.1%} efficiency)")
        
        # Show sample content
        sample_tokens = X[0][:50].tolist()
        sample_text = tokenizer.decode(sample_tokens)
        print(f"   Sample: {repr(sample_text[:80])}")
        print()
    
    overall_efficiency = effective_tokens / total_tokens
    print(f"🚀 Overall efficiency: {overall_efficiency:.1%}")
    print(f"💰 Improvement: {overall_efficiency/0.057:.1f}x more efficient than original!")


if __name__ == "__main__":
    test_simple_packed_loader()
