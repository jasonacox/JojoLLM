#!/usr/bin/python3
"""
Jojo LLM Text Generation Utility

Python script to load a GPT model checkpoint and generate text.

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import sys
import os
import time
import datetime
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import argparse
import signal
import readline  # For better command line editing in interactive mode

# Import extended tokenizer support
from setup_tokenizer import get_extended_tokenizer, format_user_message, format_system_message, format_assistant_message, SPECIAL_TOKENS

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT model.")
    parser.add_argument('checkpoint', nargs='?', default='models/story5000.pt', 
                       help='Model checkpoint filename (default: models/story5000.pt)')
    parser.add_argument('--nonstop', action='store_true', 
                       help='Generate text continuously (nonstop mode)')
    parser.add_argument('--prompt', type=str, default='\n', 
                       help='Starting prompt for generation (default: newline)')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='Load prompt from file')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive mode for continued prompts')
    parser.add_argument('--chat', action='store_true',
                       help='Enter interactive chat mode with ChatML formatting')
    parser.add_argument('--system', type=str, default='', 
                       help='System message for chat mode (default: none)')
    parser.add_argument('--use_base_tokenizer', action='store_true',
                       help='Use base GPT-2 tokenizer instead of extended tokenizer (not recommended for chat)')
    parser.add_argument('--seed', type=int, default=1334, 
                       help='Random seed (default: 1334)')
    parser.add_argument('--temp', type=float, default=1.0, 
                       help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max_tokens', type=int, default=500, 
                       help='Maximum number of tokens to generate (default: 500)')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (options: cuda[:id], cpu, mps, auto[default])')
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                       choices=['float32', 'bfloat16', 'float16'], 
                       help='Data type for inference (default: bfloat16)')
    parser.add_argument('--top_k', type=int, default=200, 
                       help='Top-k sampling (default: 200)')
    parser.add_argument('--no_delay', action='store_true',
                       help='Disable delay between tokens for faster generation')
    parser.add_argument('--output', type=str, default=None,
                       help='Save generated text to a file (provide filename)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed information including token IDs during generation')
    return parser.parse_args()

def setup_device(device_arg):
    """Set up and select the appropriate device."""
    # Default to auto-detection
    if device_arg is None:
        if torch.cuda.is_available():
            # Show CUDA device information and prompt for selection
            print(f"{BLUE}CUDA devices available:{ENDC}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # Free memory in GB
                mem_total = props.total_memory / (1024 ** 3)  # Total memory in GB
                print(f"  [{i}] {props.name} - Free: {mem_free:.2f} GB / Total: {mem_total:.2f} GB")
                
            while True:
                try:
                    user_input = input(f"{BLUE}Select CUDA device [0-{torch.cuda.device_count()-1}] (default: 0): {ENDC}")
                    if not user_input:
                        selected_device = 0
                        break
                    selected_device = int(user_input)
                    if 0 <= selected_device < torch.cuda.device_count():
                        break
                    else:
                        print(f"{YELLOW}Invalid device index.{ENDC}")
                except ValueError:
                    print(f"{YELLOW}Please enter a valid integer.{ENDC}")
                except KeyboardInterrupt:
                    print(f"\n{YELLOW}User interrupted. Exiting.{ENDC}")
                    sys.exit(0)
                    
            return f'cuda:{selected_device}', 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'mps'
        else:
            print(f"{YELLOW}No GPU detected, using CPU.{ENDC}")
            return 'cpu', 'cpu'
    
    # Parse user-specified device
    if device_arg.startswith('cuda'):
        if ':' in device_arg:
            device_idx = int(device_arg.split(':')[1])
            if device_idx >= torch.cuda.device_count():
                print(f"{YELLOW}Warning: CUDA device {device_idx} not found. Using device 0.{ENDC}")
                return 'cuda:0', 'cuda'
        elif not torch.cuda.is_available():
            print(f"{YELLOW}CUDA not available. Falling back to CPU.{ENDC}")
            return 'cpu', 'cpu'
        return device_arg, 'cuda'
    elif device_arg == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', 'mps'
    else:
        if device_arg != 'cpu':
            print(f"{YELLOW}Device '{device_arg}' not recognized or not available. Using CPU.{ENDC}")
        return 'cpu', 'cpu'

def setup_model(checkpoint_file, device, seed, dtype, verbose=False):
    """Set up the model and prepare for inference."""
    print(f"{BLUE}Loading Checkpoint:{ENDC} {checkpoint_file}")
    
    # Setup random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup device and precision
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device.startswith('cpu') else torch.amp.autocast(device_type=device.split(':')[0], dtype=ptdtype)
    
    if verbose:
        print(f"{BLUE}Using device:{ENDC} {device} with {dtype}")
    
    # Load model
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        if 'model_args' not in checkpoint:
            print(f"{RED}Error: Invalid checkpoint format. Missing 'model_args'.{ENDC}")
            sys.exit(1)
            
        gptconf = GPTConfig(**checkpoint['model_args'])
        if verbose:
            print(f"{YELLOW}{gptconf}{ENDC}")
        else:
            # Show simplified configuration
            print(f"{YELLOW}Model configuration:{ENDC} layers={gptconf.n_layer}, heads={gptconf.n_head}, embedding={gptconf.n_embd}")
        
        model = GPT(gptconf)
        
        if 'model' not in checkpoint:
            print(f"{RED}Error: Invalid checkpoint format. Missing 'model' weights.{ENDC}")
            sys.exit(1)
            
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        # Try to load state dict with error handling for potential mismatches
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"{RED}Error loading model state:{ENDC} {str(e)}")
            print(f"{YELLOW}This could be due to a version mismatch between the checkpoint and the current model.{ENDC}")
            sys.exit(1)
            
        model.eval()
        model.to(device)
        print(f"{BLUE}Number of parameters:{ENDC} {model.get_num_params()/1e6:.2f}M")
        
        return model, ctx, device
    except FileNotFoundError:
        print(f"{RED}Error: Model file '{checkpoint_file}' not found.{ENDC}")
        print(f"{YELLOW}Make sure you've trained a model and the path is correct.{ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Error loading model:{ENDC} {str(e)}")
        sys.exit(1)

def load_prompt(args):
    """Load prompt from arguments or file."""
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            print(f"{BLUE}Loaded prompt from:{ENDC} {args.prompt_file}")
            return prompt
        except Exception as e:
            print(f"{RED}Error loading prompt file:{ENDC} {str(e)}")
            print(f"{YELLOW}Falling back to default prompt.{ENDC}")
    
    return args.prompt

def process_prompt(prompt, device, use_extended=True):
    """Process the input prompt."""
    # Use the extended tokenizer if requested and available
    if use_extended:
        try:
            enc = get_extended_tokenizer()
            print(f"{BLUE}Using extended tokenizer with special tokens{ENDC}")
        except Exception as e:
            print(f"{YELLOW}Warning: Could not initialize extended tokenizer: {str(e)}{ENDC}")
            print(f"{YELLOW}Falling back to standard GPT-2 tokenizer{ENDC}")
            enc = tiktoken.get_encoding("gpt2")
    else:
        enc = tiktoken.get_encoding("gpt2")
    
    print(f"{BLUE}Prompt:{ENDC} {repr(prompt)}")
    try:
        # Always use allowed_special="all" to handle special tokens
        start_ids = enc.encode(prompt, allowed_special="all")
            
        if len(start_ids) == 0:
            print(f"{YELLOW}Warning: Empty prompt after encoding. Adding a newline character.{ENDC}")
            start_ids = enc.encode("\n", allowed_special="all")
            
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        return x, start_ids, enc
    except Exception as e:
        print(f"{RED}Error processing prompt:{ENDC} {str(e)}")
        sys.exit(1)

def process_token(token, enc, nonstop, verbose=False):
    """Process a single token, returning text to display or None to stop."""
    # Define a list of known special tokens and their names
    special_token_names = {
        50256: "<|endoftext|>",  # Standard GPT-2 EOT token
        50257: "<|im_start|>",   # ChatML format start tag
        50258: "<|im_end|>"      # ChatML format end tag
    }
    
    # Handle invalid tokens (ensure we're in a valid range)
    if not isinstance(token, int) or token < 0:
        print(f"{YELLOW}Warning: Invalid token type or value: {type(token)}, {token}{ENDC}")
        return ""
    
    # Ensure encoder is properly initialized
    if enc is None:
        print(f"{YELLOW}Warning: Encoder is None, attempting to initialize standard GPT-2 tokenizer{ENDC}")
        try:
            enc = tiktoken.get_encoding("gpt2")
        except Exception as e:
            print(f"{RED}Error initializing fallback tokenizer: {str(e)}{ENDC}")
            return ""
    
    # Skip tokens outside vocabulary range
    try:
        n_vocab = enc.n_vocab
    except AttributeError:
        # If n_vocab is not available, try a reasonable default for GPT-2
        print(f"{YELLOW}Warning: Could not access encoder vocabulary size. Using default GPT-2 size.{ENDC}")
        n_vocab = 50257  # Base GPT-2 vocabulary size plus one for special cases
    
    if token >= n_vocab:
        print(f"{YELLOW}Warning: Token ID {token} is outside vocabulary range (max: {n_vocab-1}){ENDC}")
        return ""
    
    # Handle end-of-text token
    eot_token = getattr(enc, 'eot_token', 50256)  # Default to standard GPT-2 EOT if not defined
    
    if token == eot_token:
        if verbose:
            print(f"\n{BLUE}[TOKEN {token} = <|endoftext|>]{ENDC}")
        print(f"\n{GREEN}--- The End ---{ENDC}\n")
        if not nonstop:
            return None
        return "\n"
    
    try:
        # Safely decode the token
        text = enc.decode([token])
        
        # In verbose mode, print token value alongside the text
        if verbose:
            # Check if this is a known special token
            if token in special_token_names:
                print(f"{BLUE}[TOKEN {token} = {special_token_names[token]}]{ENDC} ", end='', flush=True)
            # Check if it's likely a special token (by ID range or content)
            elif token >= 50256 or (text.strip() and ('<|' in text or '|>' in text)):
                token_repr = repr(text).replace("'", "")
                print(f"{BLUE}[TOKEN {token} = {token_repr}]{ENDC} ", end='', flush=True)
            # Handle whitespace tokens
            elif text.strip() == '':
                token_repr = repr(text).replace("'", "")
                if token_repr == "\\n":
                    print(f"{BLUE}[{token}:⏎]{ENDC}", end='', flush=True)
                elif token_repr == " ":
                    print(f"{BLUE}[{token}:␣]{ENDC}", end='', flush=True)
                else:
                    print(f"{BLUE}[{token}:{token_repr}]{ENDC}", end='', flush=True)
            # Regular tokens
            else:
                print(f"{BLUE}[{token}]{ENDC}", end='', flush=True)
        
        return text
    except Exception as e:
        print(f"{YELLOW}Warning: Error decoding token {token}: {str(e)}{ENDC}")
        return ""

def print_token(text, delay, verbose=False):
    """Print a token with appropriate formatting."""
    if text == '\n':
        print()
    else:
        print(text, end='', flush=True)
        if delay > 0 and not verbose:  # Skip delay in verbose mode to make debugging faster
            time.sleep(delay)

def generate_text(model, x, start_ids, enc, args, ctx):
    """Generate text based on prompt and parameters."""
    output_text = ""
    buffer = ""  # Buffer to collect special tokens
    in_special_token = False
    
    print("\n" + "=" * 40 + "\n")
    print(f"{GREEN}Generating text (max tokens: {'∞' if args.nonstop else args.max_tokens}, "
          f"temp: {args.temp}, top_k: {args.top_k}{', verbose mode' if args.verbose else ''})...{ENDC}\n")
    
    delay = 0 if args.no_delay else 0.005
    token_count = 0
    
    # Handle graceful interruption
    original_sigint = signal.getsignal(signal.SIGINT)
    
    def signal_handler(sig, frame):
        print(f"\n\n{YELLOW}** User Break **{ENDC}")
        signal.signal(signal.SIGINT, original_sigint)
        # Return control to the main program
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Special token detection helpers
    def is_start_of_special_token(text):
        return text == '<' or text == '<|' or text == '<|i' or text == '<|im' or text == '<|im_'
    
    def is_special_token(text):
        return text in ['<|im_start|>', '<|im_end|>', '<|endoftext|>'] or text.startswith('<|im_start|>') or text.startswith('<|im_end|>')
    
    def process_and_print_token(text, delay, verbose, in_buffer=False):
        """Process token text and print it appropriately"""
        nonlocal buffer, in_special_token, output_text
        
        # If we're in verbose mode, always print and add to output
        if verbose:
            output_text += text
            print_token(text, delay, verbose)
            return
            
        # In normal mode, handle special tokens
        if in_buffer:
            buffer += text
            
            # Check if we have a complete special token
            if is_special_token(buffer):
                # We found a complete special token - add to output but don't print
                output_text += buffer
                
                # Check if there's text after the special token (e.g., role name)
                if buffer == '<|im_start|>' and len(text) > len('<|im_start|>'):
                    # Extract and handle any content after the special token
                    extra = text[len('<|im_start|>'):]
                    if extra.strip():
                        # Don't print role names in normal mode
                        if extra.strip() in ['user', 'assistant', 'system']:
                            output_text += extra
                        else:
                            output_text += extra
                            print_token(extra, delay, verbose)
                
                # Reset buffer
                buffer = ""
                in_special_token = False
            return
            
        # Check if this could be the start of a special token
        if is_start_of_special_token(text):
            buffer = text
            in_special_token = True
            return
            
        # Regular text, print and add to output
        output_text += text
        print_token(text, delay, verbose)
    
    try:
        with torch.no_grad():
            with ctx:
                if args.nonstop:
                    # Streaming mode - in nonstop mode we don't set a max_tokens limit
                    # unless the user explicitly sets one even with nonstop flag
                    max_tokens = args.max_tokens if args.max_tokens != 500 else None
                    stream = model.generate_stream(x, temperature=args.temp, top_k=args.top_k, max_tokens=max_tokens)
                    
                    if args.verbose:
                        print(f"\n{BLUE}Token sequence (original prompt tokens: {len(start_ids)}):{ENDC}")
                        print(f"{BLUE}Format: [token_id] or [token_id:repr] for whitespace or [TOKEN token_id = repr] for special tokens{ENDC}\n")
                        
                    for _, new_token in stream:
                        token_count += 1
                        
                        # Handle token
                        text = process_token(new_token, enc, args.nonstop, args.verbose)
                        if text is None:
                            break
                        if text:
                            if args.verbose:
                                output_text += text
                                print_token(text, delay, args.verbose)
                            else:
                                # In normal mode, handle special tokens carefully
                                if in_special_token:
                                    process_and_print_token(text, delay, args.verbose, in_buffer=True)
                                else:
                                    process_and_print_token(text, delay, args.verbose)
                            
                else:
                    # Fixed token count mode
                    full_sequence = model.generate(x, args.max_tokens, temperature=args.temp, top_k=args.top_k)
                    generated_tokens = full_sequence[0, len(start_ids):].tolist()
                    token_count = len(generated_tokens)
                    
                    if args.verbose:
                        print(f"\n{BLUE}Token sequence (original prompt tokens: {len(start_ids)}):{ENDC}")
                        print(f"{BLUE}Format: [token_id] or [token_id:repr] for whitespace or [TOKEN token_id = repr] for special tokens{ENDC}\n")
                    
                    # Process all tokens
                    for new_token in generated_tokens:
                        text = process_token(new_token, enc, False, args.verbose)
                        if text is None:  # End of text token and not nonstop
                            break
                        if text:
                            if args.verbose:
                                output_text += text
                                print_token(text, delay, args.verbose)
                            else:
                                # In normal mode, handle special tokens carefully
                                if in_special_token:
                                    process_and_print_token(text, delay, args.verbose, in_buffer=True)
                                else:
                                    process_and_print_token(text, delay, args.verbose)
                
                # Check for failed generation
                if token_count == 0:
                    print(f"{YELLOW}No tokens were generated. The model may not be properly initialized "
                          "or the prompt may be too restrictive.{ENDC}")
                    
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}** Generation interrupted **{ENDC}")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n{RED}CUDA out of memory error.{ENDC} Try using a smaller model, the CPU device, or reducing batch size.")
        else:
            print(f"\n{RED}Runtime error during generation:{ENDC} {str(e)}")
    except Exception as e:
        print(f"\n{RED}Unexpected error during generation:{ENDC} {str(e)}")
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_sigint)
    
    print("\n")
    return output_text

def save_output(text, filename):
    """Save generated text to a file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"{GREEN}Output saved to:{ENDC} {filename}")
    except Exception as e:
        print(f"{RED}Error saving output:{ENDC} {str(e)}")

def interactive_mode(model, enc, args, ctx, device):
    """Run interactive mode allowing continuous prompts."""
    print(f"\n{BLUE}========================================"
          f"\n   Interactive Mode"
          f"\n   Type 'exit', 'quit', or press Ctrl+C to exit"
          f"\n========================================{ENDC}\n")
    
    while True:
        try:
            # Get user input
            prompt = input(f"{GREEN}Enter prompt:{ENDC} ")
            
            # Check for exit commands
            if prompt.lower().strip() in ['exit', 'quit', 'bye', '/exit', '/quit']:
                print(f"\n{YELLOW}Exiting interactive mode.{ENDC}")
                break
                
            if not prompt.strip():
                prompt = "\n"  # Default to newline for empty prompts
                
            # Process prompt and generate text
            x, start_ids, _ = process_prompt(prompt, device)
            output_text = generate_text(model, x, start_ids, enc, args, ctx)
            
            # Save output if requested
            if args.output:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{os.path.splitext(args.output)[0]}_{timestamp}{os.path.splitext(args.output)[1]}"
                save_output(prompt + output_text, filename)
                
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Exiting interactive mode.{ENDC}")
            break
        except Exception as e:
            print(f"{RED}Error in interactive mode:{ENDC} {str(e)}")

def format_chat_prompt(system_message, conversation=None):
    """
    Format a prompt using the ChatML format with special tokens.
    
    Args:
        system_message: A system message to set the personality/instructions for the AI
                       (if empty, no system message will be included)
        conversation: List of dictionaries with 'role' and 'content' keys
                     (role is one of 'user' or 'assistant')
    
    Returns:
        A formatted chat prompt ready for tokenization
    """
    if conversation is None:
        conversation = []
    
    # Format system message if provided
    result = []
    if system_message and system_message.strip():
        result.append(format_system_message(system_message))
    
    # Add conversation messages
    for message in conversation:
        if message['role'] == 'user':
            result.append(format_user_message(message['content']))
        elif message['role'] == 'assistant':
            result.append(format_assistant_message(message['content']))
    
    # Join all parts with newlines
    return "\n".join(result)

def interactive_chat_mode(model, enc, args, ctx, device):
    """Run interactive chat mode with proper formatting."""
    print(f"\n{BLUE}========================================"
          f"\n   Interactive Chat Mode"
          f"\n   Type 'exit', 'quit', or press Ctrl+C to exit"
          f"\n========================================{ENDC}\n")
    
    # Initialize conversation
    try:
        # Initialize the extended tokenizer for chat mode
        try:
            enc = get_extended_tokenizer()
            print(f"{BLUE}Using extended tokenizer with special tokens for chat{ENDC}")
        except Exception as e:
            print(f"{YELLOW}Warning: Could not initialize extended tokenizer: {str(e)}{ENDC}")
            print(f"{YELLOW}Falling back to standard GPT-2 tokenizer{ENDC}")
            enc = tiktoken.get_encoding("gpt2")
        
        # Use system message from command line if provided
        system_message = args.system.strip() if args.system.strip() else ""
        if system_message:
            print(f"{BLUE}System: {ENDC}{system_message}")
        
        conversation = []
        
        while True:
            # Get user input
            user_input = input(f"{GREEN}User: {ENDC}")
            
            # Check for exit commands
            if user_input.lower().strip() in ['exit', 'quit', 'bye', '/exit', '/quit']:
                print(f"\n{YELLOW}Exiting interactive chat mode.{ENDC}")
                break
                
            if not user_input.strip() and len(conversation) == 0:
                user_input = "Hello! How are you today?"
                print(f"{YELLOW}Using default greeting: '{user_input}'{ENDC}")
            elif not user_input.strip():
                continue
                
            # Add user message to conversation
            conversation.append({'role': 'user', 'content': user_input})
            
            # Format the complete prompt
            prompt = format_chat_prompt(system_message, conversation)
            
            # Add the assistant start tag to prompt response generation
            prompt += "\n<|im_start|>assistant\n"
            
            # Process prompt and generate text
            x, start_ids, _ = process_prompt(prompt, device, use_extended=True)
            
            # Create a copy of args with a lower max_tokens for chat responses
            chat_args = argparse.Namespace(**vars(args))
            if chat_args.max_tokens == 500:  # Only change if user didn't specify a custom value
                chat_args.max_tokens = 150   # Use a smaller default for chat responses
            
            print(f"{BLUE}Assistant: {ENDC}", end='', flush=True)
            output_text = generate_text(model, x, start_ids, enc, chat_args, ctx)
            
            # Clean up the response
            # Remove any incomplete special tokens
            if output_text.endswith("<|"):
                output_text = output_text[:-2].rstrip()
            elif output_text.endswith("<|im"):
                output_text = output_text[:-4].rstrip()
            elif output_text.endswith("<|im_"):
                output_text = output_text[:-5].rstrip()
            
            # Extract just the assistant's response
            if "<|im_end|>" in output_text:
                output_text = output_text.split("<|im_end|>")[0].strip()
            
            # Clean up any special tokens that might remain
            output_text = output_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
            
            # Remove repeated role patterns
            role_patterns = ["user", "assistant", "system"]
            for pattern in role_patterns:
                if output_text.startswith(pattern + "\n"):
                    output_text = output_text[len(pattern)+1:].lstrip()
            
            # Handle cases where the model starts generating a new conversation turn
            for pattern in role_patterns:
                if f"\n{pattern}\n" in output_text:
                    output_text = output_text.split(f"\n{pattern}\n")[0].strip()
            
            # Remove any lines containing only role names
            lines = output_text.split('\n')
            filtered_lines = []
            for line in lines:
                if line.strip() not in role_patterns:
                    filtered_lines.append(line)
            output_text = '\n'.join(filtered_lines)
            
            # Add assistant response to conversation history
            conversation.append({'role': 'assistant', 'content': output_text})
            
            # Save output if requested
            if args.output:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{os.path.splitext(args.output)[0]}_{timestamp}{os.path.splitext(args.output)[1]}"
                chat_log = format_chat_prompt(system_message, conversation)
                save_output(chat_log, filename)
                
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Exiting interactive chat mode.{ENDC}")
    except Exception as e:
        print(f"{RED}Error in interactive chat mode:{ENDC} {str(e)}")

def main():
    """Main function to run the generator."""
    print(f"\n{BOLD}{BLUE}========================================\n"
          "   Jojo LLM Text Generation Utility\n"
          "========================================{ENDC}\n")
    
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device, device_type = setup_device(args.device)
    
    # Setup the model
    model, ctx, device = setup_model(args.checkpoint, device, args.seed, args.dtype, args.verbose)
    
    # Determine if we should use the extended tokenizer
    use_extended = not args.use_base_tokenizer
    
    if args.chat:
        # Run interactive chat mode with special token formatting
        # We'll initialize the tokenizer inside the function
        interactive_chat_mode(model, None, args, ctx, device)
    elif args.interactive:
        # Load a base prompt
        prompt = load_prompt(args)
        # Get tokenizer
        x, start_ids, enc = process_prompt(prompt, device, use_extended=use_extended)
        # Run standard interactive mode
        interactive_mode(model, enc, args, ctx, device)
    else:
        # Standard generation mode
        prompt = load_prompt(args)
        x, start_ids, enc = process_prompt(prompt, device, use_extended=use_extended)
        output_text = generate_text(model, x, start_ids, enc, args, ctx)
        
        # Save output if requested
        if args.output:
            save_output(prompt + output_text, args.output)

if __name__ == "__main__":
    main()

