#!/usr/bin/python3
"""
Jojo LLM Tokenizer Setup Utility

This script provides functionality to set up an extended tiktoken tokenizer
with custom special tokens for the ChatML format. This implementation follows the
approach demonstrated in examples/extend-tiktoken.py, which creates a new Encoding
object with special tokens defined.

Key features:
- Extends the GPT-2 tokenizer with ChatML special tokens
- Provides utility functions for formatting chat messages
- Improves token efficiency by representing special tokens as single tokens
- Reduces token count by ~40% for conversational data

Usage:
    from setup_tokenizer import get_extended_tokenizer, format_user_message
    
    # Get the extended tokenizer
    enc = get_extended_tokenizer()
    
    # Format and encode a message
    message = format_user_message("Hello, how are you?")
    tokens = enc.encode(message, allowed_special="all")

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import tiktoken

# Define special tokens for chat format
SPECIAL_TOKENS = {
    "<|im_start|>": 50257,  # First ID after standard GPT-2 vocabulary
    "<|im_end|>": 50258,
    "<|endoftext|>": 50256  # This is already in GPT-2 but included for completeness
}

def get_extended_tokenizer(base_encoding="gpt2"):
    """
    Creates and returns an extended tiktoken tokenizer with custom special tokens.
    
    Args:
        base_encoding: The base encoding to extend (default: 'gpt2')
        
    Returns:
        A tiktoken.Encoding object with custom special tokens
    """
    # Get the base encoding
    base_enc = tiktoken.get_encoding(base_encoding)
    
    # Create a new encoding with our special tokens
    special_tokens = SPECIAL_TOKENS.copy()
    
    # If not using gpt2, adjust token IDs to be beyond the base vocabulary
    if base_encoding != "gpt2":
        base_vocab_size = base_enc.n_vocab
        special_tokens = {
            "<|im_start|>": base_vocab_size + 1,  # First ID after base vocabulary
            "<|im_end|>": base_vocab_size + 2,
            # Use the correct <|endoftext|> ID for the base encoding (if it exists)
            # or create a new one
        }
        
        # Handle base encodings with existing <|endoftext|> token
        try:
            # Try to encode the token to see if it exists
            endoftext_tokens = base_enc.encode("<|endoftext|>", allowed_special="all")
            if len(endoftext_tokens) == 1:
                # It exists as a single token, use its ID
                special_tokens["<|endoftext|>"] = endoftext_tokens[0]
            else:
                # It doesn't exist as a single token, create new ID
                special_tokens["<|endoftext|>"] = base_vocab_size + 3
        except:
            # If encoding fails, create a new ID
            special_tokens["<|endoftext|>"] = base_vocab_size + 3
    
    # Create the extended encoding
    custom_enc = tiktoken.Encoding(
        name=f"{base_encoding}_chatml",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=special_tokens
    )
    
    return custom_enc

def encode_with_extended_tokenizer(text, base_encoding="gpt2"):
    """
    Encodes text using the extended tokenizer with special token support.
    
    Args:
        text: Text to encode
        base_encoding: Base encoding to use
        
    Returns:
        List of token IDs
    """
    # Get the extended tokenizer
    enc = get_extended_tokenizer(base_encoding)
    
    # Encode with special tokens allowed
    return enc.encode(text, allowed_special="all")

def encode_with_special_tokens(text, encoding="gpt2"):
    """
    Encodes text with special tokens using the base tokenizer.
    This is a fallback for when the extended tokenizer isn't available.
    
    Args:
        text: Text to encode
        encoding: Base encoding to use
        
    Returns:
        List of token IDs
    """
    enc = tiktoken.get_encoding(encoding)
    return enc.encode(text, allowed_special="all")

def format_user_message(message):
    """Format a message from the user with special tokens"""
    return f"<|im_start|>user\n{message.strip()}\n<|im_end|>"

def format_assistant_message(message):
    """Format a message from the assistant with special tokens"""
    return f"<|im_start|>assistant\n{message.strip()}\n<|im_end|>"

def format_system_message(message):
    """Format a system message with special tokens"""
    return f"<|im_start|>system\n{message.strip()}\n<|im_end|>"

if __name__ == "__main__":
    # Example usage
    enc = get_extended_tokenizer()
    
    # Create a sample conversation
    conversation = [
        format_system_message("You are a helpful AI assistant."),
        format_user_message("Hello! How are you today?"),
        format_assistant_message("I'm doing well, thank you for asking! How can I help you today?")
    ]
    
    # Join conversation turns with an endoftext token at the end
    full_text = "\n".join(conversation) + "\n<|endoftext|>\n"
    
    print("=== Sample Conversation ===")
    print(full_text)
    print("==========================\n")
    
    # Encode with our extended tokenizer
    tokens = enc.encode(full_text, allowed_special="all")
    
    print(f"Encoded into {len(tokens)} tokens")
    print(f"Tokens: {tokens[:20]}...")
    
    # Decode back to text
    decoded = enc.decode(tokens)
    print(f"\nDecoded correctly? {decoded == full_text}")
    
    # Print token IDs for special tokens
    print("\nSpecial token IDs:")
    for token, token_id in SPECIAL_TOKENS.items():
        special_tokens = enc.encode(token, allowed_special="all")
        print(f"  {token}: {special_tokens}")
