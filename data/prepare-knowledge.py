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
    --model MODEL         Specify LLM model for answer generation or reformatting
    --reset               Delete all output and checkpoint files to start over
    --append              Reset checkpoint but append to existing output files
    --max_questions N     Limit the number of questions processed
    --verbose             Enable detailed output for troubleshooting
    
Notes:
    - The script automatically deletes the checkpoint file upon successful completion
    - Checkpoint file is only preserved if the process is interrupted or fails

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

# Constants
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
LLM_API_URL = "http://10.0.1.25:8000/v1/chat/completions"

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

def query_llm(question, api_url, api_key, model, temperature, verbose=False, squad_answer=None):
    """Queries the LLM to get an answer for a given question, or to reformat a SQuAD answer if provided."""
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
    try:
        # Add a timeout to avoid hanging if the server is not responding
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content']
        if verbose:
            print(f"  - Received answer: \"{answer[:50]}...\"")
        return answer.strip()
    except requests.exceptions.ConnectionError:
        # Connection error likely means the LLM server isn't available
        raise Exception("Could not connect to LLM API server. Please ensure the server is running.")
    except requests.exceptions.Timeout:
        raise Exception("LLM API request timed out. The server might be overloaded.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise Exception(f"LLM API endpoint not found at {api_url}. Please check the URL.")
        else:
            raise Exception(f"HTTP error from LLM API: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
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
    parser.add_argument('--llm_api_url', type=str, default=LLM_API_URL, help='URL of the LLM API endpoint.')
    parser.add_argument('--api_key', type=str, default='3-laws-safe', help='API key for the LLM.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name for the LLM.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for LLM generation.')
    parser.add_argument('--max_questions', type=int, default=None, help='Maximum number of questions to process from SQuAD.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for step-by-step details.')
    parser.add_argument('--use_squad_answers', action='store_true', help='Use answers from SQuAD dataset instead of querying LLM.')
    parser.add_argument('--reset', action='store_true', help='Delete all output and checkpoint files to start over.')
    parser.add_argument('--append', action='store_true', help='Reset checkpoint but append to existing output files instead of overwriting them.')
    return parser.parse_args()

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
        
        generated_count = 0
        llm_available = True  # Flag to track LLM availability
        
        # Open files in append mode to continue where we left off
        try:
            with open(train_file, 'a', encoding='utf-8') as f_train, \
                 open(val_file, 'a', encoding='utf-8') as f_val:
                for i, qa_pair in enumerate(tqdm(qa_to_process, desc="Generating Q&A pairs", ncols=80)):
                    current_index = start_index + i
                    question = qa_pair['question']
                    squad_answer = qa_pair['answer'] if args.use_squad_answers else None

                    # If we've already found that the LLM is not available and we need it,
                    # skip to using the squad answer directly or use a placeholder
                    if not llm_available and (not args.use_squad_answers or (args.use_squad_answers and args.model)):
                        if args.use_squad_answers:
                            answer = squad_answer
                        else:
                            # Skip this question if we can't get an answer
                            continue
                    else:
                        # Try to get an answer from the LLM if needed
                        try:
                            if args.use_squad_answers and args.model:
                                answer = query_llm(question, args.llm_api_url, args.api_key, args.model, args.temperature, args.verbose, squad_answer=squad_answer)
                            elif args.use_squad_answers:
                                answer = squad_answer
                            else:
                                answer = query_llm(question, args.llm_api_url, args.api_key, args.model, args.temperature, args.verbose)
                        except Exception as e:
                            llm_available = False
                            print(f"\nError: Could not connect to LLM API. Will use SQuAD answers directly if available.")
                            if args.use_squad_answers:
                                answer = squad_answer
                            else:
                                # Skip this question if we can't get an answer
                                continue
                    
                    if answer:
                        user_message = format_user_message(question)
                        assistant_message = format_assistant_message(answer)
                        formatted_entry = f"{user_message}\n{assistant_message}"

                        if random.random() < 0.9:
                            f_train.write(formatted_entry + CONVERSATION_SEPARATOR)
                        else:
                            f_val.write(formatted_entry + CONVERSATION_SEPARATOR)
                        generated_count += 1

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
