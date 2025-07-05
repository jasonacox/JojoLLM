#!/usr/bin/python3
"""
Jojo LLM Text Tokenizer

This script tokenizes a text file using the extended tokenizer from setup_tokenizer.py
and saves the result as a binary file (with .bin extension). It also optionally creates
additional files with token boundaries and raw tokens.

Usage:
    python tokenize.py <input_file> [options]

Options:
    --output FILE       Custom output filename (default: <input_file>.bin)
    --encoding STR      Base encoding to use (default: 'gpt2')
    --raw               Save raw tokens to .tokens.txt file
    --boundaries        Save token boundaries to .boundaries.txt file
    --standard          Use standard tiktoken instead of extended tokenizer
    --verbose           Show detailed information during tokenization

Examples:
    python tokenize.py data/input.txt
    python tokenize.py data/story.txt --raw --boundaries --verbose
    python tokenize.py data/dialogue.txt --encoding cl100k_base

Author: Jason Cox
GitHub: https://github.com/jasonacox/jojo
Date: July 3, 2025
"""

import os
import sys
import argparse
import numpy as np
import tiktoken
from pathlib import Path

# Try to import the extended tokenizer
try:
    # Add parent directory to path if script is in a subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    if os.path.basename(script_dir) != os.path.basename(parent_dir):
        sys.path.append(parent_dir)
    
    from setup_tokenizer import get_extended_tokenizer
    HAS_EXTENDED_TOKENIZER = True
except ImportError:
    HAS_EXTENDED_TOKENIZER = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Tokenize a text file and save as binary token file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tokenize.py data/input.txt
  python tokenize.py data/story.txt --raw --boundaries --verbose
  python tokenize.py data/dialogue.txt --encoding cl100k_base
  python tokenize.py data/chat.txt --standard
  python tokenize.py data/test.txt --compare
        """
    )
    
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', help='Path to the output binary file (default: <input_file>.bin)')
    parser.add_argument('--encoding', default='gpt2', help='Base encoding to use (default: gpt2)')
    parser.add_argument('--raw', action='store_true', help='Save raw tokens to .tokens.txt file')
    parser.add_argument('--boundaries', action='store_true', help='Save token boundaries to .boundaries.txt file')
    parser.add_argument('--standard', action='store_true', help='Use standard tiktoken instead of extended tokenizer')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information during tokenization')
    parser.add_argument('--compare', action='store_true', help='Compare standard and extended tokenization')
    
    return parser.parse_args()

def get_tokenizer(encoding_name, use_extended=True):
    """Get the appropriate tokenizer based on options."""
    if use_extended and HAS_EXTENDED_TOKENIZER:
        # Use extended tokenizer with special tokens support
        enc = get_extended_tokenizer(encoding_name)
        tokenizer_type = "extended"
    else:
        # Fall back to standard tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        tokenizer_type = "standard"
    
    return enc, tokenizer_type

def read_text_file(file_path):
    """Read a text file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def tokenize_text(text, tokenizer, verbose=False, allowed_special="all"):
    """Tokenize text and return token IDs."""
    try:
        if verbose:
            print(f"Tokenizing {len(text)} characters...")
        
        # Encode the text
        token_ids = tokenizer.encode(text, allowed_special=allowed_special)
        
        if verbose:
            print(f"Generated {len(token_ids)} tokens")
            
        return token_ids
    except Exception as e:
        print(f"Error during tokenization: {e}")
        sys.exit(1)

def save_token_data(token_ids, output_path, save_raw=False, save_boundaries=False, 
                   tokenizer=None, input_text=None, verbose=False):
    """Save token data to files."""
    try:
        # Convert to numpy array for binary saving
        token_array = np.array(token_ids, dtype=np.uint16)
        
        # Save the binary token file
        token_array.tofile(output_path)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if verbose:
            print(f"Saved {len(token_ids)} tokens to {output_path} ({file_size_mb:.2f} MB)")
        
        # Analyze token distribution for reporting
        token_counts = {}
        special_token_count = 0
        for token_id in token_ids:
            if token_id not in token_counts:
                token_counts[token_id] = 0
            token_counts[token_id] += 1
            
            if token_id >= 50000:  # Special tokens usually have IDs >= 50000
                special_token_count += 1
                
        unique_tokens = len(token_counts)
        
        # Save raw tokens if requested
        if save_raw and tokenizer:
            raw_path = f"{output_path}.tokens.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                # Header with stats
                f.write(f"# Token ID: token string representation\n")
                f.write(f"# Total tokens: {len(token_ids)}\n")
                f.write(f"# Unique tokens: {unique_tokens}\n")
                f.write(f"# Special tokens: {special_token_count}\n\n")
                
                for token_id in token_ids:
                    try:
                        token_bytes = tokenizer.decode_single_token_bytes(token_id)
                        token_str = token_bytes.decode('utf-8', errors='replace')
                        f.write(f"{token_id}: {token_str}\n")
                    except:
                        f.write(f"{token_id}: <special_token_{token_id}>\n")
            
            if verbose:
                print(f"Saved raw tokens to {raw_path}")
        
        # Save token boundaries if requested
        if save_boundaries and input_text:
            boundaries_path = f"{output_path}.boundaries.txt"
            
            with open(boundaries_path, 'w', encoding='utf-8') as f:
                # Write header with statistics
                f.write(f"# Token boundaries analysis\n")
                f.write(f"# Total tokens: {len(token_ids)}\n")
                f.write(f"# Unique tokens: {unique_tokens}\n")
                f.write(f"# Special tokens: {special_token_count}\n")
                f.write(f"# Tokens per character: {len(token_ids) / len(input_text):.2f}\n")
                f.write(f"# Characters per token: {len(input_text) / len(token_ids):.2f}\n\n")
                
                # First, decode the entire text to have a reference
                full_decoded = tokenizer.decode(token_ids)
                
                # Process each token to get its boundaries
                position = 0
                f.write("# Format: Token index (ID): 'token text' | occurrence count\n\n")
                
                # Track the most common tokens for statistics
                top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                
                for i, token_id in enumerate(token_ids):
                    try:
                        # Handle special tokens separately
                        if token_id >= 50000:
                            # For special tokens, just note them without trying to find boundaries
                            try:
                                # Some tokenizers have a special token name mapping
                                if hasattr(tokenizer, "decode_single_token_bytes"):
                                    token_bytes = tokenizer.decode_single_token_bytes(token_id)
                                    token_str = token_bytes.decode('utf-8', errors='replace')
                                else:
                                    # Fallback for other tokenizers
                                    token_str = f"<special_token_{token_id}>"
                            except:
                                token_str = f"<special_token_{token_id}>"
                                
                            f.write(f"Token {i} (ID {token_id}): '{token_str}' (special token) | count: {token_counts.get(token_id, 1)}\n")
                        else:
                            # For normal tokens, find their boundaries
                            # Get this specific token by decoding a small window and finding the new part
                            window_size = 1
                            
                            while True:
                                # Get a few tokens before for context
                                start_idx = max(0, i-window_size)
                                
                                # Decode chunks to find the token's text
                                if start_idx == i:
                                    # We're at the beginning of the text
                                    chunk_with = tokenizer.decode([token_ids[i]])
                                    token_text = chunk_with
                                else:
                                    chunk_before = tokenizer.decode(token_ids[start_idx:i])
                                    chunk_with = tokenizer.decode(token_ids[start_idx:i+1])
                                    
                                    # The difference is our token
                                    if len(chunk_with) > len(chunk_before):
                                        token_text = chunk_with[len(chunk_before):]
                                        break
                                    
                                # If we couldn't find a clear boundary, try with a smaller window
                                window_size -= 1
                                if window_size < 0:
                                    # If we still can't find it, just decode the single token
                                    token_text = tokenizer.decode([token_id])
                                    break
                            
                            # Write the token information
                            f.write(f"Token {i} (ID {token_id}): '{token_text}' | count: {token_counts.get(token_id, 1)}\n")
                    except Exception as e:
                        f.write(f"Token {i} (ID {token_id}): Error extracting - {str(e)}\n")
                
                # Add top token statistics at the end
                f.write("\n\n# Most common tokens:\n")
                f.write("# ID: count - token\n")
                for token_id, count in top_tokens:
                    try:
                        if hasattr(tokenizer, "decode_single_token_bytes"):
                            token_bytes = tokenizer.decode_single_token_bytes(token_id)
                            token_str = token_bytes.decode('utf-8', errors='replace')
                        else:
                            token_str = tokenizer.decode([token_id])
                    except:
                        token_str = f"<special_token_{token_id}>"
                    
                    f.write(f"# {token_id}: {count} - {token_str}\n")
            
            if verbose:
                print(f"Saved token boundaries to {boundaries_path}")
                
    except Exception as e:
        print(f"Error saving token data: {e}")
        sys.exit(1)

def analyze_tokens(token_ids):
    """Analyze token statistics."""
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    special_tokens = sum(1 for t in token_ids if t >= 50000)
    normal_tokens = total_tokens - special_tokens
    
    # Count token frequencies
    token_counts = {}
    for token_id in token_ids:
        if token_id not in token_counts:
            token_counts[token_id] = 0
        token_counts[token_id] += 1
    
    # Get most common tokens
    most_common = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total": total_tokens,
        "unique": unique_tokens,
        "special": special_tokens,
        "normal": normal_tokens,
        "most_common": most_common
    }

def compare_tokenizations(input_text, encoding_name, verbose=False):
    """Compare standard and extended tokenization."""
    print("\n=== Tokenization Comparison ===")
    
    # Get both tokenizers
    std_enc = tiktoken.get_encoding(encoding_name)
    if HAS_EXTENDED_TOKENIZER:
        ext_enc = get_extended_tokenizer(encoding_name)
    else:
        print("Extended tokenizer not available. Cannot compare.")
        return
    
    # Tokenize with both
    # Allow special tokens to avoid errors with special token text
    std_tokens = std_enc.encode(input_text, allowed_special="all")
    ext_tokens = ext_enc.encode(input_text, allowed_special="all")
    
    # Get statistics
    std_stats = analyze_tokens(std_tokens)
    ext_stats = analyze_tokens(ext_tokens)
    
    # Print comparison
    print(f"Standard tokenizer ({encoding_name}):")
    print(f"  - Total tokens: {std_stats['total']}")
    print(f"  - Unique tokens: {std_stats['unique']}")
    print(f"  - Special tokens: {std_stats['special']}")
    
    print(f"\nExtended tokenizer ({encoding_name}_chatml):")
    print(f"  - Total tokens: {ext_stats['total']}")
    print(f"  - Unique tokens: {ext_stats['unique']}")
    print(f"  - Special tokens: {ext_stats['special']}")
    
    # Calculate differences
    token_diff = std_stats['total'] - ext_stats['total']
    percent_diff = (token_diff / std_stats['total']) * 100 if std_stats['total'] > 0 else 0
    
    print(f"\nDifference: {token_diff} tokens ({percent_diff:.1f}%)")
    print(f"The {'extended' if token_diff > 0 else 'standard'} tokenizer is more efficient.")
    
    # Show some example differences if verbose
    if verbose:
        print("\nDetailed comparison of specific sections:")
        # Pick a few spots to compare in detail
        sections = [
            (0, min(10, len(std_tokens), len(ext_tokens))),  # Beginning
            (min(len(std_tokens), len(ext_tokens))//2, min(len(std_tokens), len(ext_tokens))//2 + 5),  # Middle
        ]
        
        for start, end in sections:
            if start >= len(std_tokens) or start >= len(ext_tokens):
                continue
                
            print(f"\nSection tokens {start}-{end}:")
            print("Standard:", std_tokens[start:end])
            print("Extended:", ext_tokens[start:end])
            
            print("Standard decoded:", std_enc.decode(std_tokens[start:end]))
            print("Extended decoded:", ext_enc.decode(ext_tokens[start:end]))
            
    return {
        "standard": std_stats,
        "extended": ext_stats,
        "token_diff": token_diff,
        "percent_diff": percent_diff
    }

def main():
    """Main function to run the tokenizer."""
    try:
        args = parse_args()
        
        # Determine input and output paths
        input_path = args.input_file
        if args.output:
            output_path = args.output
        else:
            # Default output is input filename with .bin extension
            output_path = f"{os.path.splitext(input_path)[0]}.bin"
        
        print(f"\n=== Jojo LLM Text Tokenizer ===")
        print(f"Input file:  {input_path}")
        print(f"Output file: {output_path}")
        
        # Verify input file exists and is readable
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' does not exist.")
            sys.exit(1)
            
        if not os.path.isfile(input_path):
            print(f"Error: '{input_path}' is not a file.")
            sys.exit(1)
            
        if not os.access(input_path, os.R_OK):
            print(f"Error: No permission to read '{input_path}'.")
            sys.exit(1)
            
        # Check if we have write permission for the output directory
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.access(output_dir, os.W_OK):
            print(f"Error: No permission to write to '{output_dir}'.")
            sys.exit(1)
        
        # Read the input file
        print(f"Reading input file...")
        input_text = read_text_file(input_path)
        file_size_kb = os.path.getsize(input_path) / 1024
        print(f"Read {len(input_text)} characters ({file_size_kb:.1f} KB)")
        
        # Check for empty file
        if not input_text:
            print("Warning: Input file is empty. Proceeding anyway.")
        
        # Special comparison mode
        if args.compare:
            compare_tokenizations(input_text, args.encoding, args.verbose)
            return
        
        # Get the tokenizer
        tokenizer, tokenizer_type = get_tokenizer(args.encoding, not args.standard)
        print(f"Using {tokenizer_type} tokenizer with {args.encoding} encoding")
        
        # Tokenize the text
        start_time = os.times()[0]  # CPU time
        token_ids = tokenize_text(input_text, tokenizer, args.verbose)
        end_time = os.times()[0]
        
        # Calculate tokenization speed
        time_taken = end_time - start_time
        tokens_per_second = len(token_ids) / time_taken if time_taken > 0 else float('inf')
        chars_per_second = len(input_text) / time_taken if time_taken > 0 else float('inf')
        
        # Analyze tokens
        stats = analyze_tokens(token_ids)
        
        # Save the token data
        save_token_data(
            token_ids, 
            output_path, 
            save_raw=args.raw, 
            save_boundaries=args.boundaries,
            tokenizer=tokenizer, 
            input_text=input_text,
            verbose=args.verbose
        )
        
        # Print summary
        print(f"\n✓ Successfully tokenized {input_path}")
        print(f"  - {len(input_text)} characters")
        print(f"  - {stats['total']} tokens ({stats['unique']} unique)")
        print(f"  - {stats['normal']} regular tokens, {stats['special']} special tokens")
        print(f"  - Compression ratio: {len(input_text) / (stats['total'] * 2):.2f} chars/byte")
        print(f"  - Processing speed: {tokens_per_second:.0f} tokens/sec, {chars_per_second:.0f} chars/sec")
        print(f"  - Output saved to {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
        
        if args.raw:
            print(f"  - Raw tokens saved to {output_path}.tokens.txt")
        if args.boundaries:
            print(f"  - Token boundaries saved to {output_path}.boundaries.txt")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
