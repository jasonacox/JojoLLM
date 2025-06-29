#!/usr/bin/python3
"""
Extended Tiktoken Tokenizer that supports adding custom special tokens
for use with the Jojo project.

HISTORICAL NOTE: This file contains the initial wrapper-based approach
that was later replaced by the direct extension method in setup_tokenizer.py.
This file is kept for reference purposes but is no longer used in the project.

This provides a wrapper around a tiktoken tokenizer that extends it
with custom special tokens, giving them dedicated token IDs. The wrapper
approach has limitations including inefficient handling of special tokens in text.
"""
import tiktoken
import regex as re

class ExtendedTokenizer:
    def __init__(self, base_encoding="gpt2", special_tokens=None):
        """
        Initialize an extended tokenizer based on a tiktoken encoding.
        
        Args:
            base_encoding: The base tiktoken encoding to use ("gpt2", "cl100k_base", etc.)
            special_tokens: Dict mapping special token strings to token IDs
        """
        # Load the base encoder
        self.base_encoder = tiktoken.get_encoding(base_encoding)
        
        # Get the vocabulary size of the base encoder
        self.base_vocab_size = self.base_encoder.n_vocab
        
        # Initialize special tokens map
        self.special_tokens = {}
        self.special_token_ids = {}
        
        # Add default special tokens that are already in the base encoder
        if base_encoding == "gpt2":
            # GPT-2 only has <|endoftext|> as a special token
            self.special_tokens["<|endoftext|>"] = 50256
            self.special_token_ids[50256] = "<|endoftext|>"
        elif base_encoding == "cl100k_base":
            # cl100k_base has more special tokens
            self.special_tokens["<|endoftext|>"] = 100257
            self.special_token_ids[100257] = "<|endoftext|>"
            self.special_tokens["<|endofprompt|>"] = 100276
            self.special_token_ids[100276] = "<|endofprompt|>"
        
        # Add custom special tokens if provided
        if special_tokens:
            for token, token_id in special_tokens.items():
                if token_id in self.special_token_ids:
                    raise ValueError(f"Token ID {token_id} is already used for {self.special_token_ids[token_id]}")
                self.special_tokens[token] = token_id
                self.special_token_ids[token_id] = token
        
        # Create a regex pattern to find special tokens in text
        self.special_token_pattern = '|'.join(
            re.escape(token) for token in sorted(self.special_tokens.keys(), key=len, reverse=True)
        )
        if self.special_token_pattern:
            self.special_token_regex = re.compile(f"({self.special_token_pattern})")
        else:
            self.special_token_regex = None
    
    def encode(self, text, allowed_special="all"):
        """
        Encode text, handling special tokens.
        
        Args:
            text: Text to encode
            allowed_special: Which special tokens are allowed ("all", a set of tokens, or None)
        
        Returns:
            List of token IDs
        """
        if not self.special_token_regex or allowed_special is None:
            # If no special tokens or they're not allowed, use base encoder directly
            return self.base_encoder.encode(text, allowed_special=allowed_special)
        
        # Handle all special tokens
        if allowed_special == "all":
            allowed_special = set(self.special_tokens.keys())
        else:
            allowed_special = set(allowed_special)
        
        # Split text by special tokens
        result = []
        segments = self.special_token_regex.split(text)
        
        for segment in segments:
            if segment in self.special_tokens and segment in allowed_special:
                # Handle special token
                result.append(self.special_tokens[segment])
            else:
                # Encode normal text with base encoder
                # Disallow base special tokens to avoid conflicts
                result.extend(
                    self.base_encoder.encode(segment, allowed_special=())
                )
        
        return result
    
    def decode(self, token_ids):
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
        
        Returns:
            Decoded text
        """
        result = []
        normal_tokens = []
        
        for token_id in token_ids:
            if token_id in self.special_token_ids:
                # Flush any accumulated normal tokens
                if normal_tokens:
                    result.append(self.base_encoder.decode(normal_tokens))
                    normal_tokens = []
                
                # Add the special token text
                result.append(self.special_token_ids[token_id])
            else:
                # Accumulate normal tokens
                normal_tokens.append(token_id)
        
        # Flush any remaining normal tokens
        if normal_tokens:
            result.append(self.base_encoder.decode(normal_tokens))
        
        return "".join(result)

# Example usage:
if __name__ == "__main__":
    # Define our ChatML special tokens with IDs starting after base vocab
    special_tokens = {
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
        # We don't need to add <|endoftext|> as it's already in the base tokenizer
    }
    
    # Create extended tokenizer
    tokenizer = ExtendedTokenizer("gpt2", special_tokens)
    
    # Test text with special tokens
    test_text = "<|im_start|>user\nHello world!\n<|im_end|>\n<|endoftext|>"
    
    # Encode and decode
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded_text}")
    print(f"Match: {test_text == decoded_text}")
    
    # Compare with base tokenizer
    base_enc = tiktoken.get_encoding("gpt2")
    base_tokens = base_enc.encode(test_text, allowed_special="all")
    print(f"\nBase tokenizer token count: {len(base_tokens)}")
    print(f"Extended tokenizer token count: {len(token_ids)}")
    print(f"Token savings: {len(base_tokens) - len(token_ids)} tokens")
