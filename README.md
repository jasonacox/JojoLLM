# Jojo: Building an LLM from Scratch

This project aims to build a large language model (LLM) from scratch, inspired by the GPT-2 architecture. We will begin training on the TinyStories dataset and gradually expand to include LLM-generated educational content. Over time, the model will be enhanced to better understand language and interaction.

## Goals

- Implement a GPT-2 style model in PyTorch.
- Start with the TinyStories dataset for initial training.
- Incrementally add more diverse and educational data.
- Grow the model’s capabilities for language understanding and interaction.

## File Overview

- `data/prepare-story.py`: Prepare the TinyStories dataset for training (download, tokenize and convert to binary format).
- `train.py`: Create the model from scratch and train it on the dataset.
- `model.py`: Defines the layers of the GPT model used in this project.
- `gen.py`: Generate output from the model based on an input.

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
   Run the following command to download, tokenize and prepare the TinyStories dataset for training:
   ```bash
   python data/prepare-story.py
   ```

4. **Train the model:**
   Run the training script. If you have a CUDA-capable GPU, you will be prompted to select which GPU to use.
   ```bash
   python train.py [--dataset DATASET] [--max_iters MAX_ITERS]
   ```
   
   Available options:
   - `--dataset`: Choose which dataset to use for training (default: 'story')
     - `story`: The TinyStories dataset for simple stories
     - `dailydialog`: The DailyDialog dataset for conversational training
     - `chat`: Human-Assistant formatted conversations for assistant-like interactions
     - `chitchat`: Simple greetings and short exchanges for basic social interactions
   - `--max_iters`: Set the total number of training iterations (default: 5000)
   - `--seed`: Set random seed for reproducibility (default: 1337)
   
   Examples:
   ```bash
   # Train on TinyStories dataset (default)
   python train.py
   
   # Train on DailyDialog dataset for conversational abilities
   python train.py --dataset dailydialog
   
   # Train on the chitchat dataset for basic greeting responses
   python train.py --dataset chitchat
   
   # Train for more iterations
   python train.py --max_iters 10000
   ```
   
   - The script will display all available CUDA devices and their memory.
   - Enter the device number you wish to use when prompted.

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

## Notes

- Checkpoints are saved automatically during training and on interruption or error.
- You can resume training from a checkpoint by modifying the script (resume logic placeholder is present).
- All dependencies are listed in `requirements.txt`.