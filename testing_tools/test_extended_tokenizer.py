#!/usr/bin/python3
"""
Jojo LLM Extended Tokenizer Test

Test script for the extended tokenizer implementation with special tokens.
This script validates that our extended tokenizer works correctly and
provides the expected token efficiency benefits.

Key tests:
- Special token encoding and decoding
- Conversation formatting and tokenization
- Token efficiency comparison
- Round-trip conversion validation

Usage:
    python test_extended_tokenizer.py
    
    # This will:
    # 1. Test special token handling
    # 2. Compare standard vs extended tokenizer behavior
    # 3. Verify token count reduction (~40% for conversations)
    # 4. Confirm proper round-trip encoding/decoding

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import tiktoken
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message, format_system_message

def test_special_tokens():
    """Test special token encoding/decoding."""
    print("\n=== Testing Special Token Handling ===\n")
    
    # Create the extended encoder with our special tokens
    from setup_tokenizer import SPECIAL_TOKENS
    
    # Get the base GPT-2 encoding
    base_enc = tiktoken.get_encoding("gpt2")
    
    # Create a new Encoding with custom special tokens
    custom_enc = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=SPECIAL_TOKENS
    )
    
    # Define special tokens to test
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
    ]
    
    print("Special tokens encoding comparison:")
    for token in special_tokens:
        # Try base encoder with and without allowed_special
        try:
            base_ids_regular = base_enc.encode(token)
            base_result1 = f"Base (no allowed_special): {base_ids_regular} - Unexpected success!"
        except:
            base_result1 = "Base (no allowed_special): Error - Expected behavior"
            
        base_ids_special = base_enc.encode(token, allowed_special="all")
        base_result2 = f"Base (allowed_special='all'): {base_ids_special}"
        
        # Try extended encoder
        ext_ids = custom_enc.encode(token, allowed_special="all")
        ext_result = f"Extended: {ext_ids}"
        
        print(f"\n{token}:")
        print(f"  {base_result1}")
        print(f"  {base_result2}")
        print(f"  {ext_result}")

def test_conversation():
    """Test a full conversation with formatting."""
    print("\n=== Testing Conversation Formatting ===\n")
    
    # Get base tokenizer
    base_enc = tiktoken.get_encoding("gpt2")
    
    # Create the extended encoder with our special tokens
    from setup_tokenizer import SPECIAL_TOKENS
    custom_enc = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=SPECIAL_TOKENS
    )
    
    # Create a sample conversation
    conversation = [
        format_system_message("You are a helpful AI assistant."),
        format_user_message("Hello! How are you today?"),
        format_assistant_message("I'm doing well, thank you for asking! How can I help you today?")
    ]
    
    # Join conversation turns
    full_text = "\n".join(conversation) + "\n<|endoftext|>\n"
    
    print("=== Sample Conversation ===")
    print(full_text)
    print("==========================\n")
    
    # Compare token counts
    base_tokens = base_enc.encode(full_text, allowed_special="all")
    ext_tokens = custom_enc.encode(full_text, allowed_special="all")
    
    print(f"Base encoder token count: {len(base_tokens)}")
    print(f"Extended encoder token count: {len(ext_tokens)}")
    
    if len(ext_tokens) < len(base_tokens):
        savings = (1 - len(ext_tokens) / len(base_tokens)) * 100
        print(f"Extended tokenizer saves {savings:.2f}% tokens!")
    else:
        print(f"No token savings with extended tokenizer.")
    
    # Test round-trip conversion
    base_decoded = base_enc.decode(base_tokens)
    ext_decoded = custom_enc.decode(ext_tokens)
    
    print("\nRound-trip conversion test:")
    print(f"Base encoder preserves text: {base_decoded == full_text}")
    print(f"Extended encoder preserves text: {ext_decoded == full_text}")

if __name__ == "__main__":
    print("\n=== Jojo Extended Tokenizer Test ===\n")
    test_special_tokens()
    test_conversation()
