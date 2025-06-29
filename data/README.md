# Jojo LLM - Data Factory

This directory contains training data used to school Jojo and scripts for preparing various datasets.

## Available Datasets

### TinyStories
- `TinyStoriesV2-GPT4-train.txt` - Training data from the TinyStories dataset
- `TinyStoriesV2-GPT4-valid.txt` - Validation data from the TinyStories dataset
- `story-train.bin` - Tokenized training data in binary format
- `story-val.bin` - Tokenized validation data in binary format

### DailyDialog
- `dailydialog-train.txt` - Training data from the DailyDialog dataset
- `dailydialog-val.txt` - Validation data from the DailyDialog dataset
- `dailydialog-train.bin` - Tokenized training data in binary format
- `dailydialog-val.bin` - Tokenized validation data in binary format

### Human-Assistant Chat Template
- `chat-train.txt` - Training data formatted with the Human-Assistant chat template
- `chat-val.txt` - Validation data formatted with the Human-Assistant chat template
- `chat-train.bin` - Tokenized training data in binary format
- `chat-val.bin` - Tokenized validation data in binary format

### Simple Chit-Chat
- `chitchat-train.txt` - Training data with simple greeting and basic conversation exchanges
- `chitchat-val.txt` - Validation data for simple chit-chat
- `chitchat-train.bin` - Tokenized training data in binary format
- `chitchat-val.bin` - Tokenized validation data in binary format

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
This script creates a specialized Human-Assistant chat format dataset from the DailyDialog dataset. It formats conversations with clear "Human:" and "Assistant:" prefixes to train the model for assistant-like interactions.

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

## Adding New Datasets

To add a new dataset, create a new preparation script following the pattern of `prepare-story.py`.
Make sure to:
1. Download the data from a reliable source
2. Properly handle text encoding
3. Tokenize the text with the appropriate tokenizer
4. Save the tokenized data in binary format

