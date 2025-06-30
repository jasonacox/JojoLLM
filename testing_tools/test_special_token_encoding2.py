#!/usr/bin/python3
"""
Test script to check if ChatML special tokens are handled as complete tokens
rather than being broken down into individual components.

This script tests how the tokenizer encodes special tokens like <|im_start|>
and checks if they are preserved as a single token or split into components.
"""
import tiktoken
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message

def test_token_encoding():
    """Test how special tokens are encoded"""
    print("\n=== Testing Special Token Encoding ===\n")
    
    # Create sample text with special tokens
    text_with_special = "<|im_start|>user\nHello!\n<|im_end|>"
    
    print(f"Test text: {text_with_special}")
    print("\n1. Standard GPT-2 tokenizer:")
    
    # Standard GPT-2 tokenizer
    std_enc = tiktoken.get_encoding("gpt2")
    
    try:
        # Without allowed_special
        std_tokens1 = std_enc.encode(text_with_special)
        std_token_strings1 = [std_enc.decode([t]) for t in std_tokens1]
        print(f"  Without allowed_special: {std_tokens1}")
        print(f"  Token strings: {std_token_strings1}")
        print(f"  This is likely breaking the special tokens into components")
    except Exception as e:
        print(f"  Error (expected): {str(e)}")
    
    # With allowed_special
    std_tokens2 = std_enc.encode(text_with_special, allowed_special="all")
    std_token_strings2 = [std_enc.decode([t]) for t in std_tokens2]
    print(f"\n  With allowed_special='all': {std_tokens2}")
    print(f"  Token strings: {std_token_strings2}")
    print(f"  Notice if '<|im_start|>' is still broken into multiple tokens")
    
    print("\n2. Extended tokenizer:")
    
    # Extended tokenizer
    ext_enc = get_extended_tokenizer()
    ext_tokens = ext_enc.encode(text_with_special, allowed_special="all")
    
    # Try to decode individual tokens to see what they represent
    ext_token_strings = []
    for t in ext_tokens:
        try:
            ext_token_strings.append(ext_enc.decode([t]))
        except:
            ext_token_strings.append(f"[ID:{t}]")
    
    print(f"  Tokens: {ext_tokens}")
    print(f"  Token strings: {ext_token_strings}")
    
    # Test the ChatML format function
    print("\n3. Testing format_user_message function:")
    
    user_msg = format_user_message("Hello, how are you?")
    print(f"  Formatted message: {user_msg}")
    
    # Encode with standard tokenizer
    std_tokens_msg = std_enc.encode(user_msg, allowed_special="all")
    std_token_strings_msg = [std_enc.decode([t]) for t in std_tokens_msg]
    print(f"  Standard tokenizer: {std_tokens_msg}")
    print(f"  Token strings: {std_token_strings_msg}")
    
    # Encode with extended tokenizer
    ext_tokens_msg = ext_enc.encode(user_msg, allowed_special="all")
    ext_token_strings_msg = []
    for t in ext_tokens_msg:
        try:
            ext_token_strings_msg.append(ext_enc.decode([t]))
        except:
            ext_token_strings_msg.append(f"[ID:{t}]")
    
    print(f"  Extended tokenizer: {ext_tokens_msg}")
    print(f"  Token strings: {ext_token_strings_msg}")
    
    # Print the custom encoder's special token mapping
    print("\n4. Extended tokenizer special tokens:")
    for name, token_id in zip(["<|im_start|>", "<|im_end|>", "<|endoftext|>"], [50257, 50258, 50256]):
        token_ids = ext_enc.encode(name, allowed_special="all")
        print(f"  {name}: {token_ids} (expected single token ID)")

if __name__ == "__main__":
    test_token_encoding()
