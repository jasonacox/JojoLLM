#!/usr/bin/python3
"""
Test the improved conversation handling with partial consumption
"""

def test_improved_packing():
    """Test that conversations are consumed completely without waste"""
    
    # Test with synthetic data including long conversations
    test_conversations = [
        [1, 2, 3],                    # Conv 0: 3 tokens (fits in block_size=10)
        [4, 5, 6, 7, 8, 9, 10, 11],  # Conv 1: 8 tokens (fits in block_size=10)
        [12, 13],                     # Conv 2: 2 tokens (fits in block_size=10)
        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], # Conv 3: 15 tokens (longer than block_size=10, would be truncated to 10)
        [29, 30, 31],                 # Conv 4: 3 tokens (fits in block_size=10)
    ]
    
    # Simulate truncation during loading (block_size = 10)
    block_size = 10
    truncated_conversations = []
    for conv in test_conversations:
        if len(conv) > block_size:
            truncated_conversations.append(conv[:block_size])
            print(f"Truncated conversation {conv} to {conv[:block_size]}")
        else:
            truncated_conversations.append(conv)
    
    print("🧪 Testing improved packing algorithm...")
    print(f"📊 Test conversations (after truncation): {truncated_conversations}")
    print(f"📊 Block size: {block_size}")
    
    # Simulate the improved packing algorithm
    shuffled_conversations = truncated_conversations.copy()  # No shuffle for predictable test
    conv_index = 0
    conv_position = 0
    
    print(f"\n📊 Packing sequences:")
    
    all_tokens_used = []
    
    for seq_num in range(5):  # Pack 5 sequences
        sequence = []
        sequence_tokens = []
        
        print(f"\n   Sequence {seq_num + 1}:")
        
        while len(sequence) < block_size:
            # Get current conversation
            current_conv = shuffled_conversations[conv_index % len(shuffled_conversations)]
            remaining_conv = current_conv[conv_position:]
            
            space_left = block_size - len(sequence)
            
            print(f"     Conv {conv_index % len(shuffled_conversations)} from pos {conv_position}: {remaining_conv} (space_left: {space_left})")
            
            if len(remaining_conv) <= space_left:
                # Remaining conversation fits completely
                sequence.extend(remaining_conv)
                sequence_tokens.extend(remaining_conv)
                all_tokens_used.extend(remaining_conv)
                print(f"       → Added complete remaining conv, sequence: {sequence}")
                # Move to next conversation
                conv_index += 1
                conv_position = 0
            else:
                # Take what we can fit
                partial = remaining_conv[:space_left]
                sequence.extend(partial)
                sequence_tokens.extend(partial)
                all_tokens_used.extend(partial)
                conv_position += space_left
                print(f"       → Added partial conv {partial}, sequence: {sequence}")
                print(f"       → Updated position to {conv_position} in conv {conv_index % len(shuffled_conversations)}")
                break
        
        print(f"     Final sequence: {sequence} (length: {len(sequence)})")
        print(f"     Tokens from this sequence: {sequence_tokens}")
    
    # Analysis
    print(f"\n📈 Analysis:")
    all_original_tokens = [token for conv in truncated_conversations for token in conv]
    print(f"   All tokens in dataset: {all_original_tokens}")
    print(f"   All tokens used: {all_tokens_used}")
    print(f"   Total tokens in dataset: {len(all_original_tokens)}")
    print(f"   Total tokens used: {len(all_tokens_used)}")
    
    # Check if we used all tokens
    if set(all_tokens_used) == set(all_original_tokens) and len(all_tokens_used) <= len(all_original_tokens):
        print(f"   ✅ Perfect: All dataset tokens used efficiently")
    else:
        missing = set(all_original_tokens) - set(all_tokens_used)
        extra = set(all_tokens_used) - set(all_original_tokens)
        if missing:
            print(f"   ⚠️  Missing tokens: {missing}")
        if extra:
            print(f"   ⚠️  Extra tokens: {extra}")
    
    print(f"\n✅ Improved packing test completed!")

if __name__ == "__main__":
    test_improved_packing()
