#!/usr/bin/env python3
"""Verify that token 0 is indeed the exclamation mark in GPT-2"""

import tiktoken

def check_token_zero():
    """Check what token 0 represents in various encodings"""
    encodings = ["gpt2", "cl100k_base", "r50k_base", "p50k_base"]
    
    for encoding_name in encodings:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            token_0_char = enc.decode([0])
            exclamation_tokens = enc.encode("!")
            
            print(f"\n{encoding_name}:")
            print(f"  Token 0 decodes to: {repr(token_0_char)}")
            print(f"  '!' encodes to: {exclamation_tokens}")
            print(f"  Token 0 is '!': {token_0_char == '!'}")
        except Exception as e:
            print(f"{encoding_name}: Error - {e}")

def check_our_tokenizer():
    """Check our extended tokenizer"""
    from setup_tokenizer import get_extended_tokenizer
    
    enc = get_extended_tokenizer()
    token_0_char = enc.decode([0])
    exclamation_tokens = enc.encode("!")
    
    print(f"\nOur extended tokenizer:")
    print(f"  Token 0 decodes to: {repr(token_0_char)}")
    print(f"  '!' encodes to: {exclamation_tokens}")
    print(f"  Token 0 is '!': {token_0_char == '!'}")

if __name__ == "__main__":
    print("🔍 Investigating Token 0 in Different Tokenizers")
    print("=" * 55)
    
    check_token_zero()
    check_our_tokenizer()
