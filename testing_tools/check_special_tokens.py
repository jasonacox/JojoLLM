#!/usr/bin/python3
"""
Jojo Special Token Test Script

This script tests how tiktoken handles the special tokens we're using in our ChatML format.
"""
import tiktoken
import json

# Special tokens for chat format
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "endoftext": "<|endoftext|>"
}

def format_user_message(message):
    """Format a message from the user with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['user']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_assistant_message(message):
    """Format a message from the assistant with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_system_message(message):
    """Format a system message with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['system']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def main():
    print("\n=== Testing Special Token Handling in tiktoken ===\n")
    
    # Create a sample conversation
    conversation = [
        format_system_message("You are a helpful AI assistant."),
        format_user_message("Hello! How are you today?"),
        format_assistant_message("I'm doing well, thank you for asking! How can I help you today?")
    ]
    
    # Join conversation turns
    full_text = "\n".join(conversation) + f"\n{SPECIAL_TOKENS['endoftext']}\n"
    
    print("=== Sample Conversation ===")
    print(full_text)
    print("==========================\n")

    # Encode with tiktoken gpt2
    enc = tiktoken.get_encoding("gpt2")
    
    # Test 1: Default encoding with allowed_special=None
    print("=== Test 1: Default encoding with allowed_special=None ===")
    try:
        tokens1 = enc.encode(full_text)
        print(f"Encoded successfully without allowed_special. Got {len(tokens1)} tokens.")
        print(f"First 10 tokens: {tokens1[:10]}")
        print(f"Tokens for '<|im_start|>': {enc.encode('<|im_start|>')}")
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
    print()
    
    # Test 2: Encoding with allowed_special="all"
    print("=== Test 2: Encoding with allowed_special='all' ===")
    try:
        tokens2 = enc.encode(full_text, allowed_special="all")
        print(f"Encoded successfully with allowed_special='all'. Got {len(tokens2)} tokens.")
        print(f"First 10 tokens: {tokens2[:10]}")
        
        # Check if <|endoftext|> is encoded properly
        endoftext_tokens = enc.encode("<|endoftext|>", allowed_special="all")
        print(f"Token ID for '<|endoftext|>': {endoftext_tokens}")
        
        # Check other special tokens
        im_start_tokens = enc.encode("<|im_start|>", allowed_special="all")
        im_end_tokens = enc.encode("<|im_end|>", allowed_special="all")
        print(f"Token ID for '<|im_start|>': {im_start_tokens}")
        print(f"Token ID for '<|im_end|>': {im_end_tokens}")
        
        # Check if we can decode back
        decoded_text = enc.decode(tokens2)
        print(f"\nCan decode back to original? {decoded_text == full_text}")
        if decoded_text != full_text:
            print("First 100 chars of decoded text:")
            print(f"  {repr(decoded_text[:100])}")
            print("First 100 chars of original text:")
            print(f"  {repr(full_text[:100])}")
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
    print()
    
    # Test 3: Checking gpt2 special tokens
    print("=== Test 3: Checking gpt2 tokenizer special tokens ===")
    try:
        # Check if known special tokens exist in the tokenizer
        print(f"Special tokens in GPT-2 tokenizer:")
        for token in ["<|endoftext|>", "<|startoftext|>", "<|pad|>"]:
            try:
                token_id = enc.encode(token, allowed_special="all")
                print(f"  {token}: {token_id}")
            except:
                print(f"  {token}: Not recognized")
    except Exception as e:
        print(f"Error checking special tokens: {str(e)}")
    
    # Test 4: Check if our special tokens are already in the vocabulary
    print("\n=== Test 4: Check if our special tokens are in the vocabulary ===")
    for token_name, token_value in SPECIAL_TOKENS.items():
        try:
            if token_value != "<|endoftext|>":  # We already know this one exists
                token_id = enc.encode(token_value, allowed_special={token_value})
                print(f"  {token_value}: {token_id}")
        except Exception as e:
            print(f"  {token_value}: Error - {str(e)}")

if __name__ == "__main__":
    main()
