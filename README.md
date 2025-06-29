# Jojo: Building an LLM from Scratch

This project aims to build a large language model (LLM) from scratch, inspired by the GPT-2 architecture. We will begin training on the TinyStories dataset and gradually expand to include LLM-generated educational content. Over time, the model will be enhanced to better understand language and interaction.

## Goals

- Implement a GPT-2 style model in PyTorch.
- Start with the TinyStories dataset for initial training.
- Incrementally add more diverse and educational data.
- Grow the model’s capabilities for language understanding and interaction.

## File Overview

- `prepare.py`: Prepare a dataset for training (tokenize and split into train vs eval).
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
   Run the following command to tokenize and split your dataset for training:
   ```bash
   python prepare.py
   ```

4. **Train the model:**
   Run the training script. If you have a CUDA-capable GPU, you will be prompted to select which GPU to use.
   ```bash
   python train.py
   ```
   - The script will display all available CUDA devices and their memory.
   - Enter the device number you wish to use when prompted.

5. **Generate text:**
   After training, you can generate text using your trained model:
   ```bash
   python gen.py
   ```

## Notes

- Checkpoints are saved automatically during training and on interruption or error.
- You can resume training from a checkpoint by modifying the script (resume logic placeholder is present).
- All dependencies are listed in `requirements.txt`.