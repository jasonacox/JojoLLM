import tiktoken

# Step 1: Get the base GPT-2 encoding
base_enc = tiktoken.get_encoding("gpt2")

# Step 2: Define your custom special tokens
custom_special_tokens = {
    "<|special1|>": 50257,
    "<|special2|>": 50258,
    "<|special3|>": 50259,
}

# Step 3: Create a new Encoding with custom special tokens
custom_enc = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=custom_special_tokens
)

# Encode with custom token support
encoded = custom_enc.encode("Hello world <|special1|> <|special2|>", allowed_special="all")
print(encoded)  # Should include 50257 and 50258

# Decode back
decoded = custom_enc.decode(encoded)
print(decoded)  # Should correctly decode including special tokens

