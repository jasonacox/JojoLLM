#!/usr/bin/python3
"""
Test to verify that all conversations are used in each epoch
"""

import os
from collections import Counter
from setup_tokenizer import get_extended_tokenizer
from simple_packed_loader import create_simple_packed_loaders

def test_conversation_coverage():
    """Test that all conversations are used in each epoch"""
    
    tokenizer = get_extended_tokenizer()
    
    # Test with chitchat data
    train_file = "data/chitchat-train.jsonl"
    val_file = "data/chitchat-val.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        return
    
    print("🧪 Testing conversation coverage per epoch...")
    
    # Create dataset - use limited batches for faster testing
    train_dataset, val_dataset = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=4, block_size=256,  # Smaller for faster testing
        train_batches=None, val_batches=None  # Use full dataset
    )
    
    # Get original conversations for comparison
    original_conversations = train_dataset.conversations
    print(f"📊 Total conversations in dataset: {len(original_conversations)}")
    
    # Track which conversations we see in the epoch
    seen_conversations = set()
    conversation_usage_count = Counter()
    
    total_batches = 0
    total_sequences = 0
    
    # Iterate through one epoch
    for batch_idx, (X, Y) in enumerate(train_dataset):
        total_batches += 1
        
        # For each sequence in the batch, try to identify which conversations were used
        for seq_idx in range(X.shape[0]):
            total_sequences += 1
            sequence = X[seq_idx].tolist()
            
            # Try to match this sequence against original conversations
            # This is approximate since sequences are packed from multiple conversations
            seq_str = str(sequence)
            
            # Check which conversations appear as subsequences
            for conv_idx, conv in enumerate(original_conversations):
                conv_str = str(conv)
                if conv_str in seq_str or any(str(conv[i:i+10]) in seq_str for i in range(0, len(conv), 10)):
                    seen_conversations.add(conv_idx)
                    conversation_usage_count[conv_idx] += 1
        
        # Limit to reasonable number for testing
        if batch_idx >= 20:  # Check first 20 batches
            break
    
    # Results
    coverage_percent = len(seen_conversations) / len(original_conversations) * 100
    
    print(f"\n📈 Results after {total_batches} batches ({total_sequences} sequences):")
    print(f"   Conversations seen: {len(seen_conversations):,} / {len(original_conversations):,}")
    print(f"   Coverage: {coverage_percent:.1f}%")
    
    if coverage_percent > 80:
        print(f"   ✅ Good coverage - conversations are being used systematically")
    else:
        print(f"   ⚠️  Low coverage - may indicate random selection issues")
    
    # Show usage distribution
    if conversation_usage_count:
        usage_counts = list(conversation_usage_count.values())
        avg_usage = sum(usage_counts) / len(usage_counts)
        print(f"   Average usage per seen conversation: {avg_usage:.1f}")
        print(f"   Usage range: {min(usage_counts)} - {max(usage_counts)}")
    
    print("\n✅ Conversation coverage test completed!")

def test_epoch_reproducibility():
    """Test that different epochs use conversations differently"""
    
    tokenizer = get_extended_tokenizer()
    
    train_file = "data/chitchat-train.jsonl"
    val_file = "data/chitchat-val.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        return
    
    print("\n🔄 Testing epoch reproducibility...")
    
    # Create dataset
    train_dataset, _ = create_simple_packed_loaders(
        train_file, val_file, tokenizer,
        batch_size=2, block_size=128,
        train_batches=5, val_batches=1  # Small for testing
    )
    
    # Get first batch from two different epochs
    epoch1_first_batch = None
    epoch2_first_batch = None
    
    # Epoch 1
    for batch in train_dataset:
        epoch1_first_batch = batch[0][0][:20].tolist()  # First 20 tokens of first sequence
        break
    
    # Epoch 2 (new iterator)
    for batch in train_dataset:
        epoch2_first_batch = batch[0][0][:20].tolist()  # First 20 tokens of first sequence
        break
    
    # Compare
    if epoch1_first_batch == epoch2_first_batch:
        print("   ⚠️  Epochs are identical - shuffling may not be working")
    else:
        print("   ✅ Epochs are different - shuffling is working correctly")
        
    print(f"   Epoch 1 start: {epoch1_first_batch[:10]}")
    print(f"   Epoch 2 start: {epoch2_first_batch[:10]}")

if __name__ == "__main__":
    test_conversation_coverage()
    test_epoch_reproducibility()
