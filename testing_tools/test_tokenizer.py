#!/usr/bin/python3
"""
Test if special tokens are being properly encoded as single tokens.
"""
import tiktoken
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message

def print_token_details(tokens, text, enc, label):
    """Print detailed information about tokens"""
    print(f"\n{label} TOKENIZER:")
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

# Get the extended tokenizer and regular tokenizer
try:
    ext_enc = get_extended_tokenizer()
    reg_enc = tiktoken.get_encoding("gpt2")
    
    print("===== TESTING TOKENIZERS =====\n")
    
    # Test special tokens
    special_token = "<|im_start|>"
    ext_tokens = ext_enc.encode(special_token, allowed_special="all")
    reg_tokens = reg_enc.encode(special_token, allowed_special="all")
    
    print_token_details(ext_tokens, special_token, ext_enc, "EXTENDED")
    print_token_details(reg_tokens, special_token, reg_enc, "REGULAR")
    
    # Test formatted message
    user_msg = format_user_message("Hello")
    ext_tokens = ext_enc.encode(user_msg, allowed_special="all")
    reg_tokens = reg_enc.encode(user_msg, allowed_special="all")
    
    print_token_details(ext_tokens, user_msg, ext_enc, "EXTENDED")
    print_token_details(reg_tokens, user_msg, reg_enc, "REGULAR")
    
except Exception as e:
    print(f"Error: {str(e)}")
