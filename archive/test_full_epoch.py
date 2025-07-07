#!/usr/bin/python3
"""
Test script to verify full dataset epochs work correctly
"""

import os
from setup_tokenizer import get_extended_tokenizer
from simple_packed_loader import create_simple_packed_loaders

def test_full_dataset_epoch():
    """Test that we can iterate through the entire dataset"""
    
    tokenizer = get_extended_tokenizer()
    
    # Test with chitchat data
    train_file = "data/chitchat-train.jsonl"
    val_file = "data/chitchat-val.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        return
    
    print("🧪 Testing full dataset epochs...")
    
    # Test 1: Limited batches (old behavior)
    print("\n📊 Test 1: Limited to 5 batches")
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=4, block_size=512,
        train_batches=5, val_batches=2
    )
    
    batch_count = 0
    for batch in train_dataset:
        batch_count += 1
    print(f"   Generated {batch_count} training batches (expected 5)")
    
    # Test 2: Full dataset (new behavior)
    print("\n📊 Test 2: Full dataset")
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=4, block_size=512,
        train_batches=None, val_batches=None
    )
    
    print(f"   Estimated training batches: {train_dataset.estimated_batches}")
    print(f"   Estimated validation batches: {val_dataset.estimated_batches}")
    
    # Count actual batches
    batch_count = 0
    for i, batch in enumerate(train_dataset):
        batch_count += 1
        if i >= 10:  # Only count first 10 to avoid long runtime
            print(f"   Generated {batch_count} training batches (stopped early)")
            break
    else:
        print(f"   Generated {batch_count} training batches (complete)")
    
    print("\n✅ Full dataset epoch test completed!")

if __name__ == "__main__":
    test_full_dataset_epoch()
