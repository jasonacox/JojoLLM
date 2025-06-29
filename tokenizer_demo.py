#!/usr/bin/python3
"""
Jojo LLM Extended Tokenizer Demo

This script demonstrates how to use the extended tokenizer with ChatML special tokens
in data preparation and text generation. It provides practical examples of how to 
integrate the extended tokenizer into various parts of the Jojo project.

Key demonstrations:
- Comparing standard vs extended tokenization
- Token efficiency analysis (~40% reduction)
- Integration examples for data preparation
- Integration examples for text generation
- Integration examples for model configuration

Usage:
    python tokenizer_demo.py
    
    # This will:
    # 1. Show tokenization comparison between standard and extended tokenizer
    # 2. Display token savings and encoding/decoding examples
    # 3. Provide code examples for integration

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import tiktoken
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message, format_system_message

def compare_tokenization():
    """Compare tokenization between standard tiktoken and extended tokenizer"""
    
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
    
    # 1. Standard GPT-2 tokenizer
    standard_enc = tiktoken.get_encoding("gpt2")
    
    # Try with and without allowed_special="all"
    try:
        standard_tokens1 = standard_enc.encode(full_text)
        print("Standard tokenizer without allowed_special:")
        print(f"  Error: Encoding succeeded but should have failed!")
    except Exception as e:
        print("Standard tokenizer without allowed_special:")
        print(f"  Error (expected): {str(e)}")
    
    standard_tokens2 = standard_enc.encode(full_text, allowed_special="all")
    print(f"\nStandard tokenizer with allowed_special='all':")
    print(f"  Tokens: {len(standard_tokens2)} tokens")
    print(f"  First 20 tokens: {standard_tokens2[:20]}")
    print(f"  '<|im_start|>' is encoded as: {standard_enc.encode('<|im_start|>', allowed_special='all')}")
    print(f"  '<|im_end|>' is encoded as: {standard_enc.encode('<|im_end|>', allowed_special='all')}")
    print(f"  '<|endoftext|>' is encoded as: {standard_enc.encode('<|endoftext|>', allowed_special='all')}")
    
    # Decoding check
    standard_decoded = standard_enc.decode(standard_tokens2)
    print(f"  Decoded correctly? {standard_decoded == full_text}")
    
    # 2. Extended tokenizer
    extended_enc = get_extended_tokenizer()
    extended_tokens = extended_enc.encode(full_text, allowed_special="all")
    print(f"\nExtended tokenizer:")
    print(f"  Tokens: {len(extended_tokens)} tokens")
    print(f"  First 20 tokens: {extended_tokens[:20]}")
    print(f"  '<|im_start|>' is encoded as: {extended_enc.encode('<|im_start|>', allowed_special='all')}")
    print(f"  '<|im_end|>' is encoded as: {extended_enc.encode('<|im_end|>', allowed_special='all')}")
    print(f"  '<|endoftext|>' is encoded as: {extended_enc.encode('<|endoftext|>', allowed_special='all')}")
    
    # Decoding check
    extended_decoded = extended_enc.decode(extended_tokens)
    print(f"  Decoded correctly? {extended_decoded == full_text}")
    
    # 3. Token efficiency comparison
    standard_token_count = len(standard_tokens2)
    extended_token_count = len(extended_tokens)
    
    print(f"\n=== Token Efficiency Comparison ===")
    print(f"Standard tokenizer: {standard_token_count} tokens")
    print(f"Extended tokenizer: {extended_token_count} tokens")
    if extended_token_count < standard_token_count:
        savings = (1 - extended_token_count / standard_token_count) * 100
        print(f"Extended tokenizer saves {savings:.2f}% tokens!")
    else:
        print(f"No token savings with extended tokenizer.")

def show_integration_examples():
    """Show examples of how to integrate the extended tokenizer in Jojo"""
    
    print("\n=== Integration Examples ===\n")
    
    print("1. In data preparation scripts (like prepare-chat.py):")
    print("""
    from setup_tokenizer import get_extended_tokenizer, format_user_message, format_assistant_message, format_system_message
    
    # Get the extended tokenizer
    enc = get_extended_tokenizer()
    
    # Format messages with special tokens
    user_turn = format_user_message("Hello, how are you?")
    assistant_turn = format_assistant_message("I'm doing well, thanks!")
    
    # When encoding for binary data files
    tokens = enc.encode(conversation_text, allowed_special="all")
    
    # Save tokens to binary file
    tokens = np.array(tokens, dtype=np.uint16)
    tokens.tofile(output_file)
    """)
    
    print("\n2. In gen.py (for inference):")
    print("""
    from setup_tokenizer import get_extended_tokenizer, format_user_message, format_system_message
    
    # Get the extended tokenizer
    enc = get_extended_tokenizer()
    
    # Format the prompt with special tokens
    system_message = format_system_message("You are a helpful assistant.")
    user_message = format_user_message(user_input)
    prompt = f"{system_message}\n{user_message}\n<|im_start|>assistant\n"
    
    # Encode the prompt for the model
    tokens = enc.encode(prompt, allowed_special="all")
    
    # Generate from the model...
    
    # Decode the result
    output_text = enc.decode(output_tokens)
    """)
    
    print("\n3. In model.py or train.py (adjusting vocabulary size):")
    print("""
    # Adjust the vocabulary size to include special tokens
    vocab_size = 50259  # GPT-2 vocab (50257) + 2 new special tokens
    
    # Update the model config
    config = GPTConfig(
        vocab_size=vocab_size,
        # ... other parameters
    )
    """)

if __name__ == "__main__":
    print("\n=== Jojo Extended Tokenizer Demo ===\n")
    compare_tokenization()
    show_integration_examples()
