#!/usr/bin/python3
"""
Jojo Tokenizer Comparison Script

This script compares how different tokenizers (gpt2 vs cl100k_base) handle our special tokens.
"""
import tiktoken
import json

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"

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

def test_tokenizer(name, conversation):
    """Test how a specific tokenizer handles our special tokens"""
    print(f"\n{BOLD}{BLUE}=== Testing {name} Tokenizer ==={ENDC}\n")
    
    enc = tiktoken.get_encoding(name)
    
    # Basic token count comparison
    print(f"{BOLD}Basic Token Count{ENDC}")
    print(f"Encoding full conversation with allowed_special='all'...")
    tokens = enc.encode(conversation, allowed_special="all")
    print(f"Total tokens: {len(tokens)}")
    
    # Test individual special tokens
    print(f"\n{BOLD}Individual Special Token Encoding{ENDC}")
    for token_name, token_value in SPECIAL_TOKENS.items():
        token_ids = enc.encode(token_value, allowed_special="all")
        print(f"  {token_value}: {token_ids} ({len(token_ids)} tokens)")
    
    # Test round-trip encoding/decoding
    print(f"\n{BOLD}Round-Trip Test{ENDC}")
    tokens = enc.encode(conversation, allowed_special="all")
    decoded = enc.decode(tokens)
    match = decoded == conversation
    print(f"Round-trip encode/decode matches: {GREEN if match else RED}{match}{ENDC}")
    
    if not match:
        print(f"{YELLOW}First difference at position: {next((i for i, (a, b) in enumerate(zip(conversation, decoded)) if a != b), min(len(conversation), len(decoded)))}{ENDC}")
        
        # Show a snippet around the first difference
        diff_pos = next((i for i, (a, b) in enumerate(zip(conversation, decoded)) if a != b), min(len(conversation), len(decoded)))
        start = max(0, diff_pos - 20)
        end = min(len(conversation), diff_pos + 20)
        print(f"Original snippet: {repr(conversation[start:end])}")
        print(f"Decoded snippet: {repr(decoded[start:end] if len(decoded) > start else 'TRUNCATED')}")
    
    # Test for native support of special tokens
    print(f"\n{BOLD}Special Token Support{ENDC}")
    # Check if tokenizer has special tokens dict
    print(f"Special tokens in vocabulary:")
    special_tokens = [
        "<|endoftext|>", 
        "<|im_start|>", 
        "<|im_end|>", 
        "<|startoftext|>",
        "<|endofprompt|>"
    ]
    for token in special_tokens:
        try:
            token_id = enc.encode(token, allowed_special="all")
            is_single = len(token_id) == 1
            color = GREEN if is_single else YELLOW
            print(f"  {token}: {color}{token_id}{ENDC} {'(single token)' if is_single else '(multiple tokens)'}")
        except Exception as e:
            print(f"  {token}: {RED}Error - {str(e)}{ENDC}")
    
    # Return the token count for comparison
    return len(tokens)

def main():
    print(f"{BOLD}{BLUE}\n=== Comparing GPT-2 vs CL100K_BASE Tokenizers ===\n{ENDC}")
    
    # Create a sample conversation
    conversation = [
        format_system_message("You are a helpful AI assistant."),
        format_user_message("Hello! How are you today?"),
        format_assistant_message("I'm doing well, thank you for asking! How can I help you today?")
    ]
    
    # Join conversation turns
    full_text = "\n".join(conversation) + f"\n{SPECIAL_TOKENS['endoftext']}\n"
    
    print(f"{BOLD}Sample Conversation:{ENDC}")
    print(f"{YELLOW}{'-' * 60}{ENDC}")
    print(full_text)
    print(f"{YELLOW}{'-' * 60}{ENDC}")
    
    # Test with gpt2 tokenizer
    gpt2_tokens = test_tokenizer("gpt2", full_text)
    
    # Test with cl100k_base tokenizer
    cl100k_tokens = test_tokenizer("cl100k_base", full_text)
    
    # Compare token counts
    print(f"\n{BOLD}{BLUE}=== Token Count Comparison ==={ENDC}")
    print(f"GPT-2 tokenizer: {gpt2_tokens} tokens")
    print(f"CL100K_BASE tokenizer: {cl100k_tokens} tokens")
    print(f"Difference: {abs(gpt2_tokens - cl100k_tokens)} tokens ({(gpt2_tokens - cl100k_tokens) / gpt2_tokens * 100:.1f}%)")
    
    if cl100k_tokens < gpt2_tokens:
        print(f"{GREEN}CL100K_BASE is more token-efficient for this content.{ENDC}")
    else:
        print(f"{YELLOW}GPT-2 is more token-efficient for this content.{ENDC}")

if __name__ == "__main__":
    main()
