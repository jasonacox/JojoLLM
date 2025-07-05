# Jojo: Building an LLM from Scratch

This project aims to build a large language model (LLM) from scratch, inspired by the GPT-2 architecture. We will begin training on the TinyStories dataset and gradually expand to include LLM-generated educational content. Over time, the model will be enhanced to better understand language and interaction.

<img width="569" alt="image" src="https://github.com/user-attachments/assets/30891367-de3a-4244-a1a2-80c4a899e949" />

## Goals

- Implement a GPT-2 style model in PyTorch.
- Start with the TinyStories dataset for initial training.
- Incrementally add more diverse and educational data.
- Grow the model’s capabilities for language understanding and interaction.

## Setup

1. **Create a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   Run the following command to tokenize and prepare all datasets for training using the extended tokenizer:
   ```bash
   python prepare.py
   ```
   
   You can also process a specific dataset (e.g., chat, chitchat, or story):
   ```bash
   python prepare.py --dataset chat
   ```
   
   Available datasets:
   - `chat`: Human-Assistant formatted conversations for assistant-like interactions
   - `chitchat`: Simple greetings and short exchanges for basic interactions
   - `story`: The TinyStories dataset for simple stories
   - `knowledge`: General knowledge Q&A pairs for factual responses
   - `dictionary`: Word definitions for vocabulary and language understanding
   
   This will process all relevant .txt files in the data/ directory and create corresponding .bin files for efficient training.

4. **Train the model:**
   Run the training script. If you have a CUDA-capable GPU, you will be prompted to select which GPU to use.
   ```bash
   python train.py [--dataset DATASET] [--epochs EPOCHS] [options]
   ```
   
   Available options:
   - `--dataset`: Choose which dataset to use for training (default: 'story')
     - `story`: The TinyStories dataset for simple stories
     - `dailydialog`: The DailyDialog dataset for conversational training
     - `chat`: Human-Assistant formatted conversations for assistant-like interactions
     - `chitchat`: Simple greetings and short exchanges for basic social interactions
     - `knowledge`: General knowledge Q&A pairs for factual responses
     - `dictionary`: Word definitions for vocabulary and language understanding
   - `--epochs`: Number of epochs to train for (default: 1)
   - `--seed`: Set random seed for reproducibility (default: 1337)
   - `--eval_interval`: How often to evaluate during an epoch (as a percentage, default: 10)
   - `--log_interval`: How often to log training progress (as a percentage, default: 10)
   - `--no-color`: Disable colored output
   - `--checkpoint`: Path to a checkpoint file to continue training from
   - `--output_checkpoint`: Custom path to save the output checkpoint (defaults to models/{dataset}_epoch{epochs}.pt)
   - `--reset_epoch`: Reset epoch counter to 0 when loading from checkpoint
   
   Examples:
   ```bash
   # Train on TinyStories dataset (default)
   python train.py
   
   # Train on DailyDialog dataset for conversational abilities
   python train.py --dataset dailydialog
   
   # Train on the chitchat dataset for basic greeting responses
   python train.py --dataset chitchat
   
   # Train for multiple epochs
   python train.py --epochs 5
   
   # Train with a specific dataset
   python train.py --dataset chat --epochs 3
   
   # Continue training from a checkpoint with a different dataset
   python train.py --checkpoint models/story_epoch5.pt --dataset chat
   
   # Continue training but reset the epoch counter to 0
   python train.py --checkpoint models/story_epoch5.pt --dataset chat --reset_epoch
   
   # Specify a custom output checkpoint path
   python train.py --dataset chat --output_checkpoint models/custom_name.pt
   ```
   
   - The script will display all available CUDA devices and their memory.
   - Enter the device number you wish to use when prompted.
   - When using an existing checkpoint, the script will check if it has already been trained beyond the requested epochs and provide options.
   - Training is epoch-based, ensuring the full dataset is used for each epoch, providing thorough and consistent coverage of your data.
   - Checkpoints are saved automatically during training at regular intervals and on interruption or error.
   - The script provides robust checkpoint handling and safe file operations to prevent checkpoint corruption.

5. **Generate text:**
   After training, you can generate text using your trained model:
   ```bash
   python gen.py [model_checkpoint] [options]
   ```

   **Generation options:**
   - Default model path is `models/story5000.pt` if not specified
   - `--nonstop`: Generate text continuously without a token limit
   - `--prompt "Your text here"`: Specify a custom starting prompt
   - `--prompt_file file.txt`: Load a prompt from a file
   - `--interactive`: Enter interactive mode for multiple prompts
   - `--seed 1234`: Set random seed for reproducible generation
   - `--temp 0.8`: Set temperature (lower = more focused, higher = more random)
   - `--max_tokens 500`: Set maximum number of tokens to generate
   - `--device cuda:0`: Specify device (cuda:N, cpu, mps, or auto-detect if not specified)
   - `--dtype float16`: Choose precision (float32, bfloat16, float16)
   - `--top_k 50`: Limit vocabulary to top K options per token
   - `--no_delay`: Disable token generation delay for faster output
   - `--output filename.txt`: Save generated text to a file
   - `--verbose`: Show detailed model information

## Special Tokens and Chat Format

This project uses the ChatML format with the following special tokens:

- `<|im_start|>user ...content... <|im_end|>` - User messages
- `<|im_start|>assistant ...content... <|im_end|>` - Assistant messages
- `<|im_start|>system ...content... <|im_end|>` - System instructions
- `<|endoftext|>` - Conversation separator

### Extended Tokenizer

The project includes an extended tokenizer that properly handles these special tokens as single tokens rather than multiple tokens. This approach:
- Improves token efficiency by ~40% for conversational data
- Helps the model better understand conversation structure
- Enables more efficient training and generation

To use the extended tokenizer:

```python
from setup_tokenizer import get_extended_tokenizer

# Get the extended tokenizer
enc = get_extended_tokenizer()

# Encode text with special tokens (always use allowed_special="all")
tokens = enc.encode("Hello <|im_start|>user text <|im_end|>", allowed_special="all")
```

The extended tokenizer is automatically used in:

- The data preparation scripts
- The generation script (gen.py) when using chat mode

### Using Chat Mode

To use the interactive chat mode with proper formatting:

```bash
python gen.py models/your_model.pt --chat
```

This will start an interactive chat session that:
- Properly formats messages using the special tokens
- Handles conversation history
- Uses the extended tokenizer for efficient token processing

For more information on implementation details, see the `setup_tokenizer.py` and
`tokenizer_demo.py` files.

<img width="569" alt="image" src="https://github.com/user-attachments/assets/30891367-de3a-4244-a1a2-80c4a899e949" />


## File Overview

- `data/prepare-story.py`: Prepare the TinyStories dataset for training (download, tokenize and convert to binary format).
- `data/prepare-chat.py`, `data/prepare-chitchat.py`: Prepare conversational datasets with ChatML formatting.
- `data/prepare-knowledge.py`: Prepares a general knowledge Q&A dataset using SQuAD and optionally a local LLM for answer generation or reformatting, with robust retry capability.
- `train.py`: Create the model from scratch and train it on the dataset.
- `model.py`: Defines the layers of the GPT model used in this project.
- `gen.py`: Generate output from the model based on an input.
- `setup_tokenizer.py`: Provides the extended tokenizer with special token support.
- `prepare.py`: Re-processes datasets using the extended tokenizer.
- `tokenizer_demo.py`: Demonstrates how to use the extended tokenizer.
- `test_extended_tokenizer.py`: Tests the extended tokenizer functionality.

## Project Structure

### Core Files
- `gen.py`: Generate output from the model based on input prompts, including interactive chat mode
- `model.py`: Defines the layers of the GPT model used in this project
- `train.py`: Create and train the model from scratch
- `setup_tokenizer.py`: Implementation of the extended tokenizer with special token support
- `prepare.py`: Utility to process datasets using the extended tokenizer
- `tokenizer_demo.py`: Demonstrates how to use the extended tokenizer

### Data Preparation
- `data/prepare-story.py`: Prepares the TinyStories dataset
- `data/prepare-chat.py`: Prepares chat-formatted conversations
- `data/prepare-chitchat.py`: Prepares simple chitchat conversations
- `data/prepare-knowledge.py`: Prepares a general knowledge Q&A dataset using SQuAD and optionally a local LLM with robust retry capability

### Testing and Development
- `testing_tools/`: Directory containing additional testing and development utilities
- `test_extended_tokenizer.py`: Test suite for the extended tokenizer
  - `check_special_tokens.py`: Tests special token handling
  - `compare_tokenizers.py`: Compares different tokenizer behaviors
  - `extended_tokenizer.py`: Initial wrapper approach (historical)
  - `quick_test.py`: Simple debugging script
  - `test_apostrophes.py`: Tests apostrophe handling in text

## Notes

- Checkpoints are saved automatically during training at regular intervals and on interruption or error.
- Checkpoint files include model weights, optimizer state, iteration count, and configuration settings.
- The training script provides robust checkpoint handling:
  - You can continue training from any checkpoint with the `--checkpoint` option
  - The script will detect if a checkpoint has already been trained beyond your requested iterations
  - You can reset the epoch counter with `--reset_epoch` to start counting from zero
  - You can specify custom output paths with `--output_checkpoint`
  - The script uses safe file operations to prevent checkpoint corruption
- All dependencies are listed in `requirements.txt`.