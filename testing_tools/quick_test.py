#!/usr/bin/python3
"""
Quick test for the tokenizer
"""
import tiktoken

print("Testing standard tiktoken first...")
enc = tiktoken.get_encoding("gpt2")
text = "Hello world <|endoftext|>"
tokens = enc.encode(text, allowed_special="all")
print(f"Encoded: {tokens}")
print(f"Decoded: {enc.decode(tokens)}")

print("\nNow testing extended tokenizer...")
base_enc = tiktoken.get_encoding("gpt2")
custom_special_tokens = {
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
    "<|endoftext|>": 50256
}
custom_enc = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=custom_special_tokens
)
text = "<|im_start|>user\nHello world\n<|im_end|>\n<|endoftext|>"
print(f"Text: {text}")
tokens = custom_enc.encode(text, allowed_special="all")
print(f"Encoded: {tokens}")
print(f"Decoded: {custom_enc.decode(tokens)}")
print(f"Match? {custom_enc.decode(tokens) == text}")

# Test single special tokens
for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    token_ids = custom_enc.encode(token, allowed_special="all")
    print(f"{token}: {token_ids}")
