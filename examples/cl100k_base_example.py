# Example update for a data preparation script
import tiktoken

# Special tokens for chat format
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "endoftext": "<|endoftext|>"
}

# Change from gpt2 to cl100k_base
enc = tiktoken.get_encoding("cl100k_base")  # Previously "gpt2"

# Process text with the new tokenizer
text = "Sample text with <|endoftext|> and <|im_start|>user..."
tokens = enc.encode(text, allowed_special="all")

# Note: The token IDs will be different, but the encoding/decoding still works correctly
decoded_text = enc.decode(tokens)
assert decoded_text == text  # This should still pass!
