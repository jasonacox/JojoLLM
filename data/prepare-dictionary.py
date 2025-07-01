#!/usr/bin/python3
"""
Jojo LLM Data Preparation - Dictionary Dataset

This script generates a dataset of dictionary entries to train the model
for understanding and defining words. It uses a word list and dictionary API
to create question-answer pairs in the format "What does [word] mean?" with
appropriate definitions as responses.

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
WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
DICTIONARY_API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"

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
    with open(file_path, 'wb') as f:
        if 'content-length' in response.headers:
            total = int(response.headers.get('content-length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, desc=file_path, ncols=80) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            f.write(response.content)
    print(f"Downloaded {file_path}.")

def get_word_definition(word, verbose=False):
    """Gets the definition of a word from the dictionary API."""
    if verbose:
        print(f"  - Looking up definition for: {word}")
    
    try:
        response = requests.get(DICTIONARY_API_URL + word, timeout=5)
        if response.status_code != 200:
            if verbose:
                print(f"  - Could not find definition for: {word}")
            return None
        
        data = response.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        # Extract the definition
        word_data = data[0]
        if "meanings" not in word_data or len(word_data["meanings"]) == 0:
            return None
        
        # Create a comprehensive definition with part of speech and examples
        definition_parts = []
        
        # Add phonetics if available
        if "phonetic" in word_data and word_data["phonetic"]:
            definition_parts.append(f"Pronunciation: {word_data['phonetic']}")
        
        # Process each meaning
        for meaning in word_data["meanings"]:
            part_of_speech = meaning.get("partOfSpeech", "")
            definitions = meaning.get("definitions", [])
            
            if definitions:
                # Add the part of speech and first definition
                main_def = definitions[0].get("definition", "")
                if main_def:
                    definition_parts.append(f"{part_of_speech.capitalize()}: {main_def}")
                
                # Add an example if available
                example = definitions[0].get("example", "")
                if example:
                    definition_parts.append(f"Example: \"{example}\"")
        
        # Add synonyms if available
        synonyms = []
        for meaning in word_data["meanings"]:
            for definition in meaning.get("definitions", []):
                if "synonyms" in definition and definition["synonyms"]:
                    synonyms.extend(definition["synonyms"][:3])  # Limit to 3 synonyms per definition
        
        if synonyms:
            # Get unique synonyms and join the first 5
            unique_synonyms = list(set(synonyms))[:5]
            definition_parts.append(f"Synonyms: {', '.join(unique_synonyms)}")
        
        # Join all parts into a complete definition
        full_definition = "\n".join(definition_parts)
        
        if verbose:
            print(f"  - Found definition: {full_definition[:50]}...")
        
        return full_definition
    
    except requests.exceptions.RequestException:
        if verbose:
            print(f"  - Error retrieving definition for: {word}")
        return None
    except Exception as e:
        if verbose:
            print(f"  - Unexpected error for {word}: {str(e)}")
        return None

# Fallback dictionary entries in case API is unavailable
DICTIONARY_DATA = [
    {
        "question": "What does 'hello' mean?",
        "answer": "Hello is an expression used as a greeting or to begin a telephone conversation. It's one of the most common greetings in the English language.\n\nExample: \"Hello! How are you today?\"\n\nSynonyms: hi, greetings, hey"
    },
    {
        "question": "Define the word 'computer'.",
        "answer": "Noun: An electronic device for storing and processing data, typically in binary form, according to instructions given to it in a variable program.\n\nExample: \"The computer crashed and I lost all my unsaved work.\"\n\nA computer typically consists of a central processing unit (CPU), memory, input/output devices, and storage components."
    },
    {
        "question": "What is the meaning of 'happy'?",
        "answer": "Adjective: Feeling or showing pleasure or contentment.\n\nExample: \"She looked happy and relaxed.\"\n\nSynonyms: joyful, cheerful, glad, delighted, pleased\n\nHappiness is an emotional state characterized by feelings of joy, contentment, and satisfaction."
    },
    {
        "question": "Define 'innovation'.",
        "answer": "Noun: The action or process of innovating; introducing new methods, ideas, or products.\n\nExample: \"Companies need innovation to survive in today's marketplace.\"\n\nSynonyms: change, alteration, revolution, transformation, creativity"
    },
    {
        "question": "What is the definition of 'ecology'?",
        "answer": "Noun: The branch of biology that deals with the relations of organisms to one another and to their physical surroundings.\n\nExample: \"The study of ecology helps us understand the impact of human activities on the environment.\"\n\nEcology examines how organisms interact with each other and with their environment."
    }
]

def format_dictionary_question(word):
    """Formats a dictionary question in different ways."""
    question_templates = [
        f"What does '{word}' mean?",
        f"Define the word '{word}'.",
        f"What is the meaning of '{word}'?",
        f"What is the definition of '{word}'?",
        f"Define '{word}'."
    ]
    return random.choice(question_templates)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare dictionary dataset using a word list and dictionary API.')
    parser.add_argument('--max_words', type=int, default=1000, help='Maximum number of words to process (default: 1000).')
    parser.add_argument('--min_word_length', type=int, default=4, help='Minimum word length to consider (default: 4).')
    parser.add_argument('--max_word_length', type=int, default=12, help='Maximum word length to consider (default: 12).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for step-by-step details.')
    parser.add_argument('--reset', action='store_true', help='Delete all output and checkpoint files to start over.')
    parser.add_argument('--sleep', type=float, default=0.5, help='Sleep time in seconds between API calls to avoid rate limiting (default: 0.5).')
    return parser.parse_args()

def main():
    """Main function to prepare the dictionary dataset."""
    print("\n=== Jojo LLM Dictionary Dataset Preparation ===\n")
    args = parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    wordlist_file = os.path.join(data_dir, "wordlist.txt")
    train_file = os.path.join(data_dir, "dictionary-train.txt")
    val_file = os.path.join(data_dir, "dictionary-val.txt")
    checkpoint_file = os.path.join(data_dir, "dictionary-checkpoint.txt")

    if args.reset:
        print("--reset specified: Deleting output and checkpoint files...")
        for f in [train_file, val_file, checkpoint_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted {f}")
        print("Reset complete. Proceeding with data preparation...")

    # Download word list
    if args.verbose:
        print("\nStep 1: Checking for word list...")
    download_if_missing(wordlist_file, WORDLIST_URL)

    # Read and filter word list
    if args.verbose:
        print("\nStep 2: Reading and filtering word list...")
    with open(wordlist_file, 'r', encoding='utf-8') as f:
        all_words = [word.strip() for word in f.readlines()]
    
    # Filter words by length and complexity
    filtered_words = [
        word for word in all_words 
        if args.min_word_length <= len(word) <= args.max_word_length 
        and word.isalpha()  # Only include alphabetic words
    ]
    
    # Shuffle the word list for randomness
    random.seed(42)
    random.shuffle(filtered_words)
    
    # Limit to max_words
    if args.max_words and args.max_words < len(filtered_words):
        filtered_words = filtered_words[:args.max_words]
    
    if args.verbose:
        print(f"- Found {len(all_words)} total words")
        print(f"- Filtered to {len(filtered_words)} words with length {args.min_word_length}-{args.max_word_length}")

    # Read checkpoint to resume
    start_index = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                start_index = int(f.read().strip())
            if args.verbose:
                print(f"\nResuming from word index {start_index}.")
        except (ValueError, IndexError):
            print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Starting from scratch.")
            start_index = 0
    
    # Filter words to process
    words_to_process = filtered_words[start_index:]
    
    if not words_to_process:
        print("No new words to process.")
    else:
        print(f"\nProcessing {len(words_to_process)} new words (from index {start_index})...")
        
        generated_count = 0
        # Open files in append mode to continue where we left off
        try:
            with open(train_file, 'a', encoding='utf-8') as f_train, \
                 open(val_file, 'a', encoding='utf-8') as f_val:
                for i, word in enumerate(tqdm(words_to_process, desc="Generating dictionary entries", ncols=80)):
                    current_index = start_index + i
                    
                    # Get definition from API
                    definition = get_word_definition(word, args.verbose)
                    
                    # Sleep to avoid rate limiting
                    time.sleep(args.sleep)
                    
                    if definition:
                        # Format the question and answer
                        question = format_dictionary_question(word)
                        answer = definition
                        
                        user_message = format_user_message(question)
                        assistant_message = format_assistant_message(answer)
                        formatted_entry = f"{user_message}\n{assistant_message}"

                        # Split between training and validation sets (90/10)
                        if random.random() < 0.9:
                            f_train.write(formatted_entry + CONVERSATION_SEPARATOR)
                        else:
                            f_val.write(formatted_entry + CONVERSATION_SEPARATOR)
                        generated_count += 1

                        # Update checkpoint after each successful word
                        with open(checkpoint_file, 'w') as f_checkpoint:
                            f_checkpoint.write(str(current_index + 1))
            
            print(f"Generated {generated_count} dictionary entries.")
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Progress saved. Exiting gracefully.")
            return
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return

    # Fallback only if train file is empty and no words were processed
    if not os.path.exists(train_file) or os.path.getsize(train_file) == 0:
        print("No words processed and no existing data. Using fallback data.")
        with open(train_file, 'w', encoding='utf-8') as f:
            fallback_data = []
            for item in DICTIONARY_DATA:
                user_message = format_user_message(item["question"])
                assistant_message = format_assistant_message(item["answer"])
                fallback_data.append(f"{user_message}\n{assistant_message}")
            f.write(CONVERSATION_SEPARATOR.join(fallback_data))

    # --- Tokenization of the final files ---
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

        train_bin = os.path.join(data_dir, 'dictionary-train.bin')
        val_bin = os.path.join(data_dir, 'dictionary-val.bin')

        train_ids.tofile(train_bin)
        if val_ids.size > 0:
            val_ids.tofile(val_bin)

        print(f"Binary files saved to:")
        print(f"  - {train_bin}")
        if val_ids.size > 0:
            print(f"  - {val_bin}")
        print("\nPreparation complete!")

    except ImportError:
        print("Could not import extended tokenizer. Skipping tokenization.")
        print("Please ensure setup_tokenizer.py is in the parent directory.")
    except Exception as e:
        print(f"An error occurred during tokenization: {e}")

if __name__ == "__main__":
    main()
