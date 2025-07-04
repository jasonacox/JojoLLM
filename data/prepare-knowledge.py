#!/usr/bin/python3
"""
Jojo LLM Data Preparation - General Knowledge Dataset

This script generates a dataset of general knowledge questions and answers
to train the model for responding to a wide range of prompts. It downloads
the SQuAD dataset for questions and uses an LLM to generate the answers.

Usage:
    python prepare-knowledge.py [options]

Options:
    --use_squad_answers   Use answers from SQuAD dataset instead of querying LLM
                          (automatically enabled if --model is not specified)
    --model MODEL         Specify LLM model for answer generation or reformatting
                          (if not provided, SQuAD answers will be used directly)
    --llm_api_url URL     URL of the LLM API endpoint (will be automatically formatted,
                          default: http://localhost:8000/v1/chat/completions)
    --llm_api_key KEY     API key for the LLM (default: 3-laws-safe)
    --temperature TEMP    Temperature for LLM generation (default: 0.7)
    --reset               Delete all output and checkpoint files to start over
    --append              Reset checkpoint but append to existing output files
    --max_questions N     Limit the number of questions processed
    --verbose             Enable detailed output for troubleshooting
    --retry_delay N       Seconds to wait between retry attempts when LLM is unavailable (default: 10)
    
Notes:
    - The script automatically deletes the checkpoint file upon successful completion
    - Checkpoint file is only preserved if the process is interrupted or fails
    - When a model is specified and LLM is unavailable, the script will retry indefinitely
      until the LLM is available again

Author: Jason A. Cox
2025 June 30
https://github.com/jasonacox/jojo
"""
import os
import random
import numpy as np
import tiktoken
from tqdm import tqdm
import requests
import json
import argparse
import time

# Constants
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_RETRY_DELAY = 10  # Default retry delay in seconds

# Special tokens for chat format
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "endoftext": "<|endoftext|>"
}
CONVERSATION_SEPARATOR = "\n<|endoftext|>\n\n"

# Helper functions for new format
def format_user_message(message):
    """Formats a message from the user with special tokens."""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['user']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_assistant_message(message):
    """Formats a message from the assistant with special tokens."""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_llm_api_url(url):
    """Ensures the LLM API URL is properly formatted with scheme and correct endpoint path.
    
    This function ensures that the URL:
    1. Has a proper scheme (http:// or https://)
    2. Ends with the correct API endpoint path (/v1/chat/completions)
    
    Args:
        url: The URL to format (e.g., "localhost:8000", "api.example.com/v1")
        
    Returns:
        A properly formatted URL that starts with http/https and ends with /v1/chat/completions
        
    Examples:
        >>> format_llm_api_url("localhost:8000")
        "http://localhost:8000/v1/chat/completions"
        
        >>> format_llm_api_url("api.example.com/v1")
        "http://api.example.com/v1/chat/completions"
        
        >>> format_llm_api_url("https://openai.com")
        "https://openai.com/v1/chat/completions"
    """
    # Add http:// if no scheme is present
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    
    # Make sure the URL ends with /v1/chat/completions
    if not url.endswith("/v1/chat/completions"):
        # Remove trailing slash if present
        if url.endswith("/"):
            url = url[:-1]
        
        # Check if URL already contains part of the endpoint
        if "/v1/chat" in url:
            # Extract base path up to /v1/chat
            base_path = url.split("/v1/chat")[0]
            url = f"{base_path}/v1/chat/completions"
        elif "/v1" in url:
            # Extract base path up to /v1
            base_path = url.split("/v1")[0]
            url = f"{base_path}/v1/chat/completions"
        else:
            # Just append the full endpoint path
            url = f"{url}/v1/chat/completions"
    
    return url

def download_if_missing(file_path, url):
    """Downloads a file from a URL if it doesn't exist."""
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    print(f"Downloading {url} to {file_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=file_path, ncols=80) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Downloaded {file_path}.")

def extract_qa_from_squad(squad_file, verbose=False):
    """Extracts all question-answer pairs from the SQuAD dataset file."""
    if verbose:
        print(f"- Reading SQuAD file: {squad_file}")
    with open(squad_file, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    qa_pairs = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible'] and qa['answers']:
                    # Use the first answer provided in the dataset
                    qa_pairs.append({"question": qa['question'], "answer": qa['answers'][0]['text']})
    if verbose:
        print(f"- Found {len(qa_pairs)} question-answer pairs. Example: '{qa_pairs[0]['question']}' -> '{qa_pairs[0]['answer']}'")
    return qa_pairs

def query_llm(question, api_url, api_key, model, temperature, verbose=False, squad_answer=None, max_retries=None, retry_delay=10):
    """Queries the LLM to get an answer for a given question, or to reformat a SQuAD answer if provided.
    
    Args:
        question: The question to ask the LLM
        api_url: URL of the LLM API
        api_key: API key for authentication
        model: LLM model to use
        temperature: Temperature parameter for generation
        verbose: Whether to print verbose logs
        squad_answer: If provided, reformat this answer instead of generating one
        max_retries: Maximum number of retries (None for infinite retries)
        retry_delay: Delay in seconds between retries
    """
    # Ensure the API URL is properly formatted
    api_url = format_llm_api_url(api_url)
    if squad_answer is not None:
        # Prompt the LLM to reformat the SQuAD answer into a complete, well-punctuated sentence
        prompt = (
            f"Question: {question}\n"
            f"Original Answer: {squad_answer}\n"
            "Rewrite the answer as a complete, well-punctuated, short sentence that directly answers the question."
        )
        user_message = prompt
    else:
        user_message = question
    if verbose:
        print(f"  - Querying LLM for: \"{user_message}\"")
    
    # Try to connect to the LLM API
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Jojo, a helpful assistant. Provide concise and accurate answers to the user's questions. Try to limit your responses to 1-2 sentences."},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature
    }
    
    retries = 0
    while max_retries is None or retries <= max_retries:
        try:
            # Add a timeout to avoid hanging if the server is not responding
            response = requests.post(api_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
            if verbose:
                print(f"  - Received answer: \"{answer[:50]}...\"")
            return answer.strip()
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, 
                requests.exceptions.HTTPError) as e:
            
            error_type = type(e).__name__
            error_message = str(e)
            
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                error_detail = f"LLM API endpoint not found at {api_url}. Please check the URL."
            elif isinstance(e, requests.exceptions.ConnectionError):
                error_detail = "Could not connect to LLM API server. Please ensure the server is running."
            elif isinstance(e, requests.exceptions.Timeout):
                error_detail = "LLM API request timed out. The server might be overloaded."
            else:
                error_detail = f"HTTP error from LLM API: {e}"
            
            retries += 1
            if max_retries is not None and retries > max_retries:
                raise Exception(f"Maximum retries ({max_retries}) exceeded. Last error: {error_detail}")
            
            print(f"\nWarning: {error_detail}")
            print(f"Retrying in {retry_delay} seconds... (attempt {retries}{' of ' + str(max_retries) if max_retries else ''})")
            
            import time
            time.sleep(retry_delay)
            
            # Let the user know we're trying again
            print(f"Retrying LLM query for: \"{user_message[:50]}...\"")
        
        except requests.exceptions.RequestException as e:
            print(f"Unrecoverable error querying LLM: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error when querying LLM: {e}")
            return None

# General Knowledge Q&A Pairs (fallback)
KNOWLEDGE_DATA = [
    {
        "question": "What is love?",
        "answer": "Love is a complex and multifaceted emotion that has been a central theme in philosophy, art, and human experience for centuries. It can refer to a deep feeling of affection and care towards another person, a strong liking for something, or a benevolent concern for the good of others. While there\'s no single definition, it often involves feelings of attachment, intimacy, and commitment."
    },
    {
        "question": "Describe photosynthesis.",
        "answer": "Photosynthesis is the process used by plants, algae, and some bacteria to convert light energy into chemical energy, through a process that converts carbon dioxide and water into glucose (a sugar) and oxygen. This process is essential for life on Earth as it produces most of the oxygen in our atmosphere and provides the primary source of energy for most ecosystems."
    },
    {
        "question": "What is the theory of relativity?",
        "answer": "The theory of relativity, developed by Albert Einstein, is actually two related theories: special relativity and general relativity. Special relativity deals with the relationship between space and time for objects moving at constant speeds. General relativity is a theory of gravitation, which describes gravity as a property of the curvature of spacetime. It\'s a cornerstone of modern physics."
    },
    {
        "question": "Explain the water cycle.",
        "answer": "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. The main stages are: evaporation (water turns into vapor and rises), condensation (vapor cools and forms clouds), precipitation (water falls back to Earth as rain, snow, etc.), and collection (water gathers in rivers, lakes, and oceans)."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial intelligence (AI) is a broad area of computer science that makes machines that can seem like they have human intelligence. It includes machine learning, where computers can learn from data without being explicitly programmed, and natural language processing, which allows computers to understand and respond to human language. AI powers many things we use daily, from recommendation engines to virtual assistants."
    }
]

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare general knowledge dataset using SQuAD and an LLM.')
    parser.add_argument('--llm_api_url', type=str, default=LLM_API_URL, 
                    help='URL of the LLM API endpoint. Will be automatically formatted to include http:// and /v1/chat/completions if needed.')
    parser.add_argument('--llm_api_key', type=str, default='3-laws-safe', help='API key for the LLM.')
    parser.add_argument('--model', type=str, default=None, help='Model name for the LLM. If not provided, SQuAD answers will be used directly.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for LLM generation.')
    parser.add_argument('--max_questions', type=int, default=None, help='Maximum number of questions to process from SQuAD.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for step-by-step details.')
    parser.add_argument('--use_squad_answers', action='store_true', help='Use answers from SQuAD dataset instead of querying LLM.')
    parser.add_argument('--reset', action='store_true', help='Delete all output and checkpoint files to start over.')
    parser.add_argument('--append', action='store_true', help='Reset checkpoint but append to existing output files instead of overwriting them.')
    parser.add_argument('--retry_delay', type=int, default=DEFAULT_RETRY_DELAY, help='Delay in seconds between retry attempts when LLM is unavailable.')
    
    args = parser.parse_args()
    # Format the LLM API URL to ensure it's properly formatted
    if args.llm_api_url:
        args.llm_api_url = format_llm_api_url(args.llm_api_url)
    return args

def main():
    """Main function to prepare the knowledge dataset."""
    print("\n=== Jojo LLM Knowledge Dataset Preparation ===\n")
    args = parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    squad_file = os.path.join(data_dir, "squad-train-v2.0.json")
    train_file = os.path.join(data_dir, "knowledge-train.txt")
    val_file = os.path.join(data_dir, "knowledge-val.txt")
    checkpoint_file = os.path.join(data_dir, "knowledge-checkpoint.txt")

    if args.reset:
        print("--reset specified: Deleting output and checkpoint files...")
        for f in [train_file, val_file, checkpoint_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted {f}")
        print("Reset complete. Proceeding with data preparation...")
    elif args.append:
        print("--append specified: Resetting checkpoint but keeping existing output files...")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Deleted {checkpoint_file}")
        print("Will continue processing from the beginning and append to existing output files.")

    # Download SQuAD dataset
    if args.verbose:
        print("\nStep 1: Checking for SQuAD dataset...")
    download_if_missing(squad_file, SQUAD_URL)

    # Extract questions from SQuAD
    if args.verbose:
        print("\nStep 2: Extracting Q&A pairs from SQuAD...")
    qa_pairs = extract_qa_from_squad(squad_file, args.verbose)
    random.seed(42)
    random.shuffle(qa_pairs)

    # Read checkpoint to resume
    start_index = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                start_index = int(f.read().strip())
            if args.verbose:
                print(f"\nResuming from question index {start_index}.")
        except (ValueError, IndexError):
            print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Starting from scratch.")
            start_index = 0
    
    # Filter questions to process
    qa_to_process = qa_pairs[start_index:]
    if args.max_questions:
        if args.verbose:
            print(f"- Limiting to {args.max_questions} new Q&A pairs.")
        qa_to_process = qa_to_process[:args.max_questions]

    if not qa_to_process:
        print("No new Q&A pairs to process.")
    else:
        print(f"\nProcessing {len(qa_to_process)} new Q&A pairs (from index {start_index})...")
        
        # If no model is provided, automatically set use_squad_answers to True
        if not args.model and not args.use_squad_answers:
            print("\nNote: No model specified, automatically using SQuAD answers directly.")
            args.use_squad_answers = True
        
        generated_count = 0
        
        # Open files in append mode to continue where we left off
        try:
            with open(train_file, 'a' if os.path.exists(train_file) and not args.reset else 'w', encoding='utf-8') as f_train, \
                 open(val_file, 'a' if os.path.exists(val_file) and not args.reset else 'w', encoding='utf-8') as f_val:
                for i, qa_pair in enumerate(tqdm(qa_to_process, desc="Generating Q&A pairs", ncols=80)):
                    current_index = start_index + i
                    question = qa_pair['question']
                    squad_answer = qa_pair['answer']

                    # Handle answer generation/reformatting with retries if model is specified
                    if args.model:
                        try:
                            if args.use_squad_answers:
                                # Reformat the SQuAD answer using LLM with infinite retries
                                answer = query_llm(
                                    question, 
                                    args.llm_api_url, 
                                    args.llm_api_key, 
                                    args.model, 
                                    args.temperature, 
                                    args.verbose, 
                                    squad_answer=squad_answer, 
                                    max_retries=None,
                                    retry_delay=args.retry_delay
                                )
                            else:
                                # Generate answer using LLM with infinite retries
                                answer = query_llm(
                                    question, 
                                    args.llm_api_url, 
                                    args.llm_api_key, 
                                    args.model, 
                                    args.temperature, 
                                    args.verbose,
                                    max_retries=None,
                                    retry_delay=args.retry_delay
                                )
                        except KeyboardInterrupt:
                            print("\nKeyboard interrupt detected. Saving checkpoint and exiting...")
                            with open(checkpoint_file, 'w') as ckpt:
                                ckpt.write(str(current_index))
                            print(f"Checkpoint saved at index {current_index}")
                            return
                    else:
                        # Use SQuAD answer directly (no model specified or explicit --use_squad_answers)
                        answer = squad_answer
                    
                    if answer:
                        user_message = format_user_message(question)
                        assistant_message = format_assistant_message(answer)
                        formatted_entry = f"{user_message}\n{assistant_message}"

                        if random.random() < 0.9:
                            f_train.write(formatted_entry + CONVERSATION_SEPARATOR)
                        else:
                            f_val.write(formatted_entry + CONVERSATION_SEPARATOR)
                        generated_count += 1

                        # Save checkpoint after each successful answer
                        with open(checkpoint_file, 'w') as f_checkpoint:
                            f_checkpoint.write(str(current_index + 1))
            
            print(f"Generated {generated_count} new knowledge conversations.")
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Progress saved. Exiting gracefully.")
            return
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return

    # Fallback only if train file is empty and no questions were processed
    if not os.path.exists(train_file) or os.path.getsize(train_file) == 0:
        print("No questions processed and no existing data. Using fallback data.")
        with open(train_file, 'w', encoding='utf-8') as f:
            fallback_data = []
            for item in KNOWLEDGE_DATA:
                user_message = format_user_message(item["question"])
                assistant_message = format_assistant_message(item["answer"])
                fallback_data.append(f"{user_message}\n{assistant_message}")
            f.write(CONVERSATION_SEPARATOR.join(fallback_data))

    # --- The rest of the script reads the final files for tokenization ---

    print("\nStep 5: Reading final datasets for tokenization...")
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data_str = f.read()
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data_str = f.read()
    except FileNotFoundError:
        print(f"Could not find {train_file} or {val_file}. Exiting tokenization.")
        return

    if not train_data_str.strip():
        print("No training data to tokenize. Exiting.")
        return

    # Encode with tiktoken
    try:
        import sys
        # Add parent directory to the path so we can import setup_tokenizer
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from setup_tokenizer import get_extended_tokenizer
        enc = get_extended_tokenizer()
        print("Using extended tokenizer with special token support...")

        train_ids = enc.encode(train_data_str, allowed_special="all")
        val_ids = enc.encode(val_data_str, allowed_special="all") if val_data_str.strip() else []

        print(f"Train data: {len(train_ids):,} tokens")
        if val_ids:
            print(f"Validation data: {len(val_ids):,} tokens")
        else:
            print("Validation data is empty.")

        # Export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16) if val_ids else np.array([], dtype=np.uint16)

        train_bin = os.path.join(data_dir, 'knowledge-train.bin')
        val_bin = os.path.join(data_dir, 'knowledge-val.bin')

        train_ids.tofile(train_bin)
        if val_ids.size > 0:
            val_ids.tofile(val_bin)

        print(f"Binary files saved to:")
        print(f"  - {train_bin}")
        if val_ids.size > 0:
            print(f"  - {val_bin}")
            
        # Delete checkpoint file on successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Deleted {checkpoint_file} (processing complete)")
            
        print("\nPreparation complete!")

    except ImportError:
        print("Could not import extended tokenizer. Skipping tokenization.")
        print("Please ensure setup_tokenizer.py is in the parent directory.")
    except Exception as e:
        print(f"An error occurred during tokenization: {e}")

if __name__ == "__main__":
    main()
