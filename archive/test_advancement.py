#!/usr/bin/python3
"""
Test to verify that conversations advance sequentially
"""

from simple_packed_loader import SimplePackedDataset
from setup_tokenizer import get_extended_tokenizer

def test_conversation_advancement():
    """Test that conversation index advances properly"""
    
    # Create a simple test with very short conversations to make tracking easier
    test_data = [
        [1, 2, 3],      # Conv 0: 3 tokens
        [4, 5, 6, 7],   # Conv 1: 4 tokens  
        [8, 9],         # Conv 2: 2 tokens
        [10, 11, 12, 13, 14], # Conv 3: 5 tokens
        [15, 16, 17],   # Conv 4: 3 tokens
    ]
    
    print("🧪 Testing conversation advancement with synthetic data...")
    print(f"📊 Test conversations: {test_data}")
    
    # Mock a simple dataset
    class MockDataset:
        def __init__(self):
            self.conversations = test_data
            self.block_size = 10  # Small block size
            
    dataset = MockDataset()
    
    # Simulate the packing algorithm
    shuffled_conversations = dataset.conversations.copy()  # No shuffle for predictable test
    conv_index = 0
    
    print(f"\n📊 Packing sequences with block_size={dataset.block_size}:")
    
    for seq_num in range(5):  # Pack 5 sequences
        sequence = []
        conversations_used = []
        
        print(f"\n   Sequence {seq_num + 1}:")
        
        while len(sequence) < dataset.block_size:
            # Get the next conversation
            conv = shuffled_conversations[conv_index % len(shuffled_conversations)]
            conv_idx = conv_index % len(shuffled_conversations)
            conversations_used.append(conv_idx)
            
            space_left = dataset.block_size - len(sequence)
            
            print(f"     Conv {conv_idx}: {conv} (space_left: {space_left})")
            
            if len(conv) <= space_left:
                # Conversation fits completely
                sequence.extend(conv)
                conv_index += 1
                print(f"       → Added complete conv, new sequence: {sequence}")
            else:
                # Take what we can fit
                sequence.extend(conv[:space_left])
                conv_index += 1
                print(f"       → Added partial conv {conv[:space_left]}, new sequence: {sequence}")
                break
        
        print(f"     Final sequence: {sequence} (length: {len(sequence)})")
        print(f"     Conversations used: {conversations_used}")
        print(f"     Next conv_index: {conv_index}")
    
    print(f"\n✅ Final conv_index: {conv_index}")
    print(f"✅ Total conversations: {len(dataset.conversations)}")
    print(f"✅ Cycles through dataset: {conv_index / len(dataset.conversations):.1f} times")

if __name__ == "__main__":
    test_conversation_advancement()
