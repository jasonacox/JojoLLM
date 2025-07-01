# Jojo LLM - Data Factory

This directory contains training data used to school Jojo and scripts for preparing various datasets.

## Available Datasets

### TinyStories 
- `TinyStoriesV2-GPT4-train.txt` - Training data from the TinyStories dataset
- `TinyStoriesV2-GPT4-valid.txt` - Validation data from the TinyStories dataset
- `story-train.bin` - Tokenized training data in binary format
- `story-val.bin` - Tokenized validation data in binary format

### DailyDialog - Chat Template
- `chat-train.txt` - Training data formatted with the Human-Assistant chat template
- `chat-val.txt` - Validation data formatted with the Human-Assistant chat template
- `chat-train.bin` - Tokenized training data in binary format
- `chat-val.bin` - Tokenized validation data in binary format

### Simple Chit-Chat - Chat Template
- `chitchat-train.txt` - Training data with simple greeting and basic conversation exchanges
- `chitchat-val.txt` - Validation data for simple chit-chat
- `chitchat-train.bin` - Tokenized training data in binary format
- `chitchat-val.bin` - Tokenized validation data in binary format

### Dictionary Dataset - Word Definitions
- `dictionary-train.txt` - Training data with word definition Q&A pairs
- `dictionary-val.txt` - Validation data for word definitions
- `dictionary-train.bin` - Tokenized training data in binary format
- `dictionary-val.bin` - Tokenized validation data in binary format

## Chat Format

This project uses the ChatML template. The tokenizer is using the tiktoken GPT-2 (r50k_base) vocabulary with an extension to include the ChatML special characters `<|im_end|>` and `<|im_start|>` (tokens 50257 and 50258).

Example:

```
<|im_start|>system
You are a helpful assistant designed to answer questions about general knowledge.
<|im_end|>
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
Paris.
<|im_end|>
<|im_start|>user
Who wrote "Romeo and Juliet"?
<|im_end|>
<|im_start|>assistant
William Shakespeare.
<|im_end|>
```

## Preparation Scripts

### prepare-story.py
This script downloads and prepares the TinyStories dataset.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-story.py
```

The script will:
1. Download the TinyStories dataset files if they don't exist
2. Tokenize the text using the GPT-2 tokenizer
3. Save the tokenized data as binary files for efficient loading

### prepare-dailydialog.py
This script downloads and prepares the DailyDialog dataset, which contains high-quality multi-turn dialogues focusing on everyday conversations. It's perfect for training models to handle small talk.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-dailydialog.py
```

The script will:
1. Download the DailyDialog dataset if it doesn't exist
2. Process the dialogues and format them for conversational training
3. Split the data into training and validation sets
4. Tokenize the text using the GPT-2 tokenizer
5. Save the tokenized data as binary files for efficient loading

### prepare-chat.py
This script downloads and prepares the DailyDialog dataset, which contains high-quality multi-turn dialogues focusing on everyday conversations. This script then creates a ChatML format dataset from the DailyDialog dataset and removes some odd puctuation spacing in that dataset.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-chat.py
```

The script will:
1. Download the DailyDialog dataset if it doesn't exist
2. Process the dialogues and reformat them using the Human-Assistant chat template
3. Refine assistant responses to be more helpful and assistant-like
4. Split the data into training and validation sets
5. Tokenize the text using the GPT-2 tokenizer
6. Save the tokenized data as binary files for efficient loading
7. Create an example chat prompt file

### prepare-chitchat.py
This script generates a simple chit-chat dataset with common greetings, questions, and short conversational exchanges. It's designed to help the model learn basic social interactions and quick responses to common queries.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-chitchat.py
```

The script will:
1. Generate a variety of greeting exchanges (like "Hi", "Hello", etc.)
2. Generate common question-answer pairs (like "How are you?")
3. Create multi-turn conversations with greetings and follow-ups
4. Split the data into training and validation sets
5. Tokenize the text using the GPT-2 tokenizer
6. Save the tokenized data as binary files for efficient loading
7. Create an example chit-chat prompt file

### prepare-knowledge.py
This script generates a general knowledge Q&A dataset using the SQuAD dataset and optionally a local LLM for answer generation or reformatting.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-knowledge.py [--use_squad_answers] [--model MODEL] [--reset] [other options]
```

Options:
- `--use_squad_answers`  Use answers from the SQuAD dataset instead of generating with the LLM. If used with `--model`, the LLM will reformat the SQuAD answer into a well-punctuated, complete sentence.
- `--model MODEL`        Specify the LLM model to use (e.g., gpt-3.5-turbo). Required for LLM-based answer generation or reformatting.
- `--reset`              Delete all output and checkpoint files before starting (fresh run).
- `--append`             Reset checkpoint but append to existing output files (useful for adding more data).
- `--max_questions N`    Limit the number of questions processed.
- `--verbose`            Enable verbose output for step-by-step details.

The script will:
1. Download the SQuAD dataset if it doesn't exist
2. Extract question-answer pairs
3. If `--use_squad_answers` is set:
   - Use SQuAD answers directly, or
   - If `--model` is also set, prompt the LLM to reformat the SQuAD answer as a complete, well-punctuated sentence
4. If `--use_squad_answers` is not set, query the LLM for each answer
5. Write Q&A pairs incrementally to output files, supporting resume after interruption
6. Tokenize the text using the GPT-2 tokenizer (with ChatML special tokens)
7. Save the tokenized data as binary files for efficient loading
8. Support `Ctrl+C` interruption and checkpointing for robust, resumable runs
9. Automatically delete checkpoint file upon successful completion

### prepare-dictionary.py
This script generates a dictionary dataset with word definitions by using a free dictionary API to look up word meanings, examples, and synonyms.

Usage:
```bash
cd /home/jason/code/jojo
python data/prepare-dictionary.py [--max_words N] [--reset] [other options]
```

Options:
- `--max_words N`        Maximum number of words to process (default: 1000).
- `--min_word_length N`  Minimum word length to consider (default: 4).
- `--max_word_length N`  Maximum word length to consider (default: 12).
- `--reset`              Delete all output and checkpoint files before starting (fresh run).
- `--verbose`            Enable verbose output for step-by-step details.
- `--sleep N`            Sleep time in seconds between API calls to avoid rate limiting (default: 0.5).

The script will:
1. Download a comprehensive English word list if it doesn't exist
2. Filter words by length and complexity
3. Look up definitions using a free dictionary API
4. Format questions like "What does X mean?" with comprehensive definitions as answers
5. Write entries incrementally to output files, supporting resume after interruption
6. Tokenize the text using the GPT-2 tokenizer (with ChatML special tokens)
7. Save the tokenized data as binary files for efficient loading
8. Support `Ctrl+C` interruption and checkpointing for robust, resumable runs

## Adding New Datasets

To add a new dataset, create a new preparation script following the pattern of `prepare-chitchat.py`.
Make sure to:
1. Download the data from a reliable source
2. Properly handle text encoding
3. Tokenize the text with the appropriate tokenizer
4. Save the tokenized data in binary format

