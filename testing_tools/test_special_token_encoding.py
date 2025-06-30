#!/usr/bin/python3
"""
Test script to check if special tokens are being properly encoded as single tokens.

This script verifies that the extended tokenizer is properly handling ChatML special tokens
by comparing token counts and examining the token IDs produced.
"""
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message
import tiktoken

def print_token_details(tokens, text, enc):
    """Print detailed information about tokens"""
    print(f"Text: {repr(text)}")
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    
    # Print each token with its decoded text
    print("Tokens breakdown:")
    for i, token in enumerate(tokens):
        try:
            token_text = enc.decode([token])
            print(f"  Token {i+1}: ID {token} → {repr(token_text)}")
        except Exception as e:
            print(f"  Token {i+1}: ID {token} → Error decoding: {str(e)}")
    print()

def test_extended_tokenizer():
    """Test the extended tokenizer with special tokens"""
    # Get the extended tokenizer and regular tokenizer
    ext_enc = get_extended_tokenizer()
    reg_enc = tiktoken.get_encoding("gpt2")
    
    print("===== TESTING EXTENDED TOKENIZER VS REGULAR TOKENIZER =====\n")
    
    # Test cases
    test_cases = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>user\nHello\n<|im_end|>",
        format_user_message("Hello"),
        format_assistant_message("Hi there!")
    ]
    
    for test_case in test_cases:
        print("=" * 80)
        print(f"Test case: {repr(test_case)}")
        print("-" * 40)
        
        # Test with extended tokenizer
        ext_tokens = ext_enc.encode(test_case, allowed_special="all")
        print("EXTENDED TOKENIZER:")
        print_token_details(ext_tokens, test_case, ext_enc)
        
        # Test with regular tokenizer
        reg_tokens = reg_enc.encode(test_case, allowed_special="all")
        print("REGULAR TOKENIZER:")
        print_token_details(reg_tokens, test_case, reg_enc)

if __name__ == "__main__":
    test_extended_tokenizer()
