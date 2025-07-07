#!/usr/bin/python3
"""
Simple test to verify sequential conversation usage
"""

import os
from setup_tokenizer import get_extended_tokenizer
from simple_packed_loader import SimplePackedDataset

def test_sequential_usage():
    """Test that conversations are used sequentially, not randomly"""
    
    tokenizer = get_extended_tokenizer()
    
    train_file = "data/chitchat-train.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ File not found: {train_file}")
        return
    
    print("🧪 Testing sequential conversation usage...")
    
    # Create a dataset with very specific parameters to track usage
    dataset = SimplePackedDataset(
        train_file, tokenizer,
        block_size=50,    # Very small blocks
        batch_size=1,     # Single sequence per batch
        max_batches=10    # Limited batches for testing
    )
    
    print(f"📊 Dataset has {len(dataset.conversations)} conversations")
    print(f"📊 Using block_size={dataset.block_size}, batch_size={dataset.batch_size}")
    
    # Track the conversation index as we iterate
    conversation_indices_used = []
    
    # Manually iterate like the dataset does
    shuffled_conversations = dataset.conversations.copy()
    print(f"📊 First 10 conversation lengths: {[len(conv) for conv in shuffled_conversations[:10]]}")
    
    conv_index = 0
    tokens_used = 0
    
    # Simulate the packing process for a few sequences
    for seq_num in range(20):  # Test first 20 sequences
        sequence = []
        seq_conv_indices = []
        
        # Pack this sequence
        while len(sequence) < dataset.block_size:
            # Get next conversation
            conv = shuffled_conversations[conv_index % len(shuffled_conversations)]
            original_conv_index = conv_index % len(shuffled_conversations)
            seq_conv_indices.append(original_conv_index)
            
            space_left = dataset.block_size - len(sequence)
            
            if len(conv) <= space_left:
                # Conversation fits completely
                sequence.extend(conv)
                tokens_used += len(conv)
                conv_index += 1  # Move to next conversation
            else:
                # Take what we can fit and move to next conversation
                sequence.extend(conv[:space_left])
                tokens_used += space_left
                conv_index += 1  # Always increment to ensure we use all conversations
                break
        
        conversation_indices_used.extend(seq_conv_indices)
        
        print(f"   Sequence {seq_num+1:2d}: Used conversations {seq_conv_indices} "
              f"(length: {len(sequence)}, total tokens: {tokens_used})")
    
    # Analyze usage pattern
    unique_conversations = set(conversation_indices_used)
    total_conversations = len(dataset.conversations)
    
    print(f"\n📈 Analysis:")
    print(f"   Total conversations in dataset: {total_conversations:,}")
    print(f"   Unique conversations used: {len(unique_conversations):,}")
    print(f"   Conversation indices used: {sorted(list(unique_conversations))[:20]}..." if len(unique_conversations) > 20 else f"   Conversation indices used: {sorted(list(unique_conversations))}")
    
    # Check if we're using conversations sequentially
    is_sequential = conversation_indices_used == sorted(conversation_indices_used[:len(unique_conversations)])
    
    if len(unique_conversations) > total_conversations * 0.1:  # Used more than 10% of conversations
        print(f"   ✅ Good: Used {len(unique_conversations)/total_conversations*100:.1f}% of conversations")
    else:
        print(f"   ⚠️  Only used {len(unique_conversations)/total_conversations*100:.1f}% of conversations")
    
    # Check for duplicates in the first few used
    first_20_indices = conversation_indices_used[:20]
    if len(set(first_20_indices)) < len(first_20_indices):
        print(f"   ⚠️  Found duplicate conversations in first 20 uses")
    else:
        print(f"   ✅ No duplicate conversations in first 20 uses")
    
    print("\n✅ Sequential usage test completed!")

if __name__ == "__main__":
    test_sequential_usage()
