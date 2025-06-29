#!/usr/bin/python3
"""
Jojo LLM Data Preparation - Simple Chit-Chat Dataset

This script creates a simple chit-chat dataset with common greetings
and short conversational exchanges to help train the model for 
basic social interactions.

Author: Jason A. Cox
2025 June 28
https://github.com/jasonacox/jojo
"""
import os
import tiktoken
import numpy as np
from tqdm import tqdm
import re
import sys
import random

# Chat format template
HUMAN_PREFIX = "Human: "
ASSISTANT_PREFIX = "Assistant: "
TURN_SEPARATOR = "\n\n"
CONVERSATION_SEPARATOR = "\n\n<|endoftext|>\n\n"

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Generate a set of common greetings, questions, and responses
def generate_chitchat_dataset():
    """
    Generate a simple dataset of common chit-chat exchanges.
    Each exchange consists of a human greeting/question and an assistant response.
    """
    chitchat_data = []
    
    # Name introduction interactions
    name_interactions = [
        # Format: (human_intro, [list_of_possible_assistant_responses])
        ("My name is Jason.", [
            "Hi Jason, it's nice to meet you! How may I help you today?",
            "Hello Jason! What can I assist you with today?",
            "Nice to meet you, Jason! How can I help you?",
            "Hi Jason! I'm Jojo. What can I do for you today?"
        ]),
        ("I am Jason.", [
            "Hello Jason! It's great to meet you. What can I help you with today?",
            "Nice to meet you, Jason. I'm Jojo. How may I assist you?",
            "Hi Jason! How can I help you today?",
            "Hello Jason! I'm Jojo, your AI assistant. What do you need help with?"
        ]),
        ("You can call me Jason.", [
            "I'll call you Jason! It's nice to meet you. How can I assist you today?",
            "Hello Jason! I'm Jojo. What can I help you with?",
            "Nice to meet you, Jason! Is there something I can help you with today?",
            "Hi Jason! What would you like help with today?"
        ])
    ]
    
    # Simple greetings with variations
    greetings = [
        # Format: (human_greeting, [list_of_possible_assistant_responses])
        ("Hi", [
            "Hello! How can I help you today?",
            "Hi there! What can I assist you with?",
            "Hello! What can I help you with?",
            "Hi! How may I assist you today?"
        ]),
        ("Hello", [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hello! How may I assist you?",
            "Hi! Is there something I can help you with?"
        ]),
        ("Hey", [
            "Hello! What can I do for you today?",
            "Hi there! How can I help?",
            "Hey there! What can I assist you with?",
            "Hello! How may I help you today?"
        ]),
        ("Good morning", [
            "Good morning! How can I assist you today?",
            "Good morning! What can I help you with?",
            "Good morning! How may I be of service today?",
            "Morning! What can I do for you today?"
        ]),
        ("Good afternoon", [
            "Good afternoon! How can I help you today?",
            "Good afternoon! What can I assist you with?",
            "Good afternoon! How may I help you?",
            "Hello there! What can I do for you this afternoon?"
        ]),
        ("Good evening", [
            "Good evening! How can I assist you today?",
            "Good evening! What can I help you with?",
            "Good evening! How may I be of service?",
            "Hello! What can I do for you this evening?"
        ])
    ]
    
    # Common questions and responses
    questions = [
        # Format: (human_question, [list_of_possible_assistant_responses])
        ("How are you?", [
            "I'm doing well, thank you for asking! How can I help you today?",
            "I'm functioning properly and ready to assist you. What can I help with?",
            "I'm good, thanks for asking! Is there something I can help you with?",
            "I'm doing great! How can I assist you today?"
        ]),
        ("What's up?", [
            "Nothing much! I'm here and ready to help you. What can I assist you with?",
            "Just waiting to help you! What can I do for you today?",
            "I'm here and ready to assist. What do you need help with?",
            "All good here! What can I help you with today?"
        ]),
        ("How's it going?", [
            "Everything's going well! How can I assist you today?",
            "Going well, thanks for asking! What can I help you with?",
            "I'm doing great and ready to help. What do you need?",
            "All good! What can I do for you today?"
        ]),
        ("What can you do?", [
            "I can answer questions, help with tasks, provide information, and engage in conversations. What would you like help with?",
            "I'm here to assist with information, answer questions, and help with various tasks. What can I help you with?",
            "I can provide information, answer questions, assist with problem-solving, and have conversations. How can I help you today?",
            "I'm designed to be helpful, informative, and conversational. What would you like assistance with?"
        ]),
        ("Who are you?", [
            "I'm an AI assistant trained to be helpful, harmless, and honest. How can I assist you today?",
            "I'm an AI assistant here to help answer your questions and assist with tasks. What can I help you with?",
            "I'm an AI designed to be helpful and provide information. How can I assist you?",
            "I'm an AI assistant trained to be helpful and informative. What can I do for you today?"
        ]),
        # Name-related questions
        ("What is your name?", [
            "My name is Jojo. How can I assist you today?",
            "I'm Jojo, your AI assistant. What can I help you with?",
            "My name is Jojo. It's nice to meet you! How can I help?",
            "I'm called Jojo. What can I do for you today?"
        ]),
        ("Who is Jojo?", [
            "That's me! I'm Jojo, an AI assistant designed to be helpful. What can I do for you today?",
            "Jojo is my name. I'm here to assist you with information and answer your questions.",
            "I'm Jojo, an AI assistant. How can I help you today?",
            "That's me - I'm Jojo, your AI assistant. How may I help you?"
        ])
    ]
    
    # Common follow-ups after initial greeting
    follow_ups = [
        # Format: (human_follow_up, [list_of_possible_assistant_responses])
        ("I need help with something", [
            "I'd be happy to help. What do you need assistance with?",
            "Sure, I'm here to help. What do you need help with specifically?",
            "Of course! Please tell me what you need help with, and I'll do my best to assist you.",
            "I'm ready to help. Could you please provide more details about what you need assistance with?"
        ]),
        ("I have a question", [
            "I'm here to answer your questions. What would you like to know?",
            "Sure, I'd be happy to answer your question. What is it?",
            "Go ahead and ask! I'll do my best to provide a helpful answer.",
            "I'm ready for your question. What would you like to know?"
        ]),
        ("Can you help me?", [
            "Absolutely! I'd be happy to help. What do you need assistance with?",
            "Yes, I'm here to help. Please let me know what you need.",
            "Of course I can help. What do you need assistance with?",
            "I'm ready to assist. What do you need help with today?"
        ]),
        ("Just saying hi", [
            "Hi there! It's nice to hear from you. Is there anything I can help you with today?",
            "Hello! Always nice to chat. Let me know if you need anything.",
            "Hi! Thanks for reaching out. I'm here if you need any assistance.",
            "Hello! Just saying hi back. Feel free to ask if you need help with anything."
        ]),
        ("Nothing much", [
            "Alright! I'm here if you need anything. Feel free to ask any questions or just chat.",
            "Sounds good. I'm available if you need assistance with anything later on.",
            "No problem. I'm here whenever you need help with something.",
            "Okay! Just let me know if there's anything I can help you with."
        ])
    ]
    
    # Generate single-turn conversations from greetings
    print("Generating greeting exchanges...")
    for greeting, responses in greetings:
        for response in responses:
            conversation = [
                f"{HUMAN_PREFIX}{greeting}",
                f"{ASSISTANT_PREFIX}{response}"
            ]
            chitchat_data.append(TURN_SEPARATOR.join(conversation))
    
    # Generate single-turn conversations from questions
    print("Generating question exchanges...")
    for question, responses in questions:
        for response in responses:
            conversation = [
                f"{HUMAN_PREFIX}{question}",
                f"{ASSISTANT_PREFIX}{response}"
            ]
            chitchat_data.append(TURN_SEPARATOR.join(conversation))
    
    # Generate two-turn conversations (greeting + follow-up)
    print("Generating multi-turn exchanges...")
    for greeting, greeting_responses in greetings:
        for greeting_response in greeting_responses[:2]:  # Limit to avoid too many combinations
            for follow_up, follow_up_responses in follow_ups:
                for follow_up_response in follow_up_responses[:1]:  # Just one response per follow-up
                    conversation = [
                        f"{HUMAN_PREFIX}{greeting}",
                        f"{ASSISTANT_PREFIX}{greeting_response}",
                        f"{HUMAN_PREFIX}{follow_up}",
                        f"{ASSISTANT_PREFIX}{follow_up_response}"
                    ]
                    chitchat_data.append(TURN_SEPARATOR.join(conversation))
    
    # Generate name introduction conversations
    print("Generating name introduction exchanges...")
    for name_intro, responses in name_interactions:
        for response in responses:
            conversation = [
                f"{HUMAN_PREFIX}{name_intro}",
                f"{ASSISTANT_PREFIX}{response}"
            ]
            chitchat_data.append(TURN_SEPARATOR.join(conversation))
    
    # Add some more complex multi-turn conversations
    
    complex_conversations = [
        # Format: [human_msg1, assistant_msg1, human_msg2, assistant_msg2, ...]
        [
            "Hi there",
            "Hello! How can I help you today?",
            "I'm just checking how you respond to simple greetings",
            "I'm designed to be friendly and helpful with all kinds of interactions, including simple greetings! Is there anything specific you'd like to know or discuss?"
        ],
        [
            "Hello, how are you?",
            "I'm doing well, thanks for asking! How can I assist you today?",
            "I'm good too, thanks for asking",
            "That's great to hear! Is there anything I can help you with today?"
        ],
        [
            "Hey there",
            "Hello! What can I do for you today?",
            "Do you have a name?",
            "My name is Jojo. I'm an AI assistant created to help you. What can I do for you today?"
        ],
        [
            "Good morning!",
            "Good morning! How can I assist you today?",
            "I'm just browsing the internet",
            "That sounds nice! If you need any information or have any questions during your browsing, feel free to ask. I'm here to help!"
        ],
        [
            "What is your name?",
            "My name is Jojo. How can I help you today?",
            "My name is Jason",
            "It's nice to meet you, Jason! How may I assist you today?"
        ],
        [
            "Hi, I'm Jason",
            "Hello Jason! It's nice to meet you. I'm Jojo, your AI assistant. How can I help you today?",
            "Can you remember my name?",
            "Yes, I'll remember that your name is Jason during our conversation. How can I assist you, Jason?"
        ]
    ]
    
    print("Adding complex conversations...")
    for conv in complex_conversations:
        formatted_conv = []
        for i, message in enumerate(conv):
            if i % 2 == 0:
                # Human message
                formatted_conv.append(f"{HUMAN_PREFIX}{message}")
            else:
                # Assistant message
                formatted_conv.append(f"{ASSISTANT_PREFIX}{message}")
        
        chitchat_data.append(TURN_SEPARATOR.join(formatted_conv))
    
    print(f"Generated {len(chitchat_data)} chit-chat conversations in total")
    return chitchat_data

def main():
    print("\n=== Jojo LLM Chit-Chat Dataset Preparation ===\n")
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(data_dir, 'chitchat')
    ensure_dir(output_base)
    
    # Generate the dataset
    chitchat_data = generate_chitchat_dataset()
    
    # Split into train and validation sets (90/10 split)
    random.seed(42)  # For reproducibility
    random.shuffle(chitchat_data)
    
    split_idx = int(0.9 * len(chitchat_data))
    train_data = chitchat_data[:split_idx]
    val_data = chitchat_data[split_idx:]
    
    print(f"Split into {len(train_data)} training and {len(val_data)} validation conversations.")
    
    # Combine all conversations into text files with conversation separators
    train_text = CONVERSATION_SEPARATOR.join(train_data)
    val_text = CONVERSATION_SEPARATOR.join(val_data)
    
    # Save the raw text files
    train_file = os.path.join(data_dir, "chitchat-train.txt")
    val_file = os.path.join(data_dir, "chitchat-val.txt")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(val_text)
        
    print(f"Raw text files saved to {train_file} and {val_file}")
    
    # Encode with tiktoken gpt2
    try:
        enc = tiktoken.get_encoding("gpt2")
        print("Encoding training data...")
        train_ids = enc.encode(train_text, allowed_special="all")
        print("Encoding validation data...")
        val_ids = enc.encode(val_text, allowed_special="all")
        
        print(f"Train data: {len(train_ids):,} tokens")
        print(f"Validation data: {len(val_ids):,} tokens")
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        sys.exit(1)

    # Export to bin files for training
    try:
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        
        train_bin = os.path.join(data_dir, 'chitchat-train.bin')
        val_bin = os.path.join(data_dir, 'chitchat-val.bin')
        
        train_ids.tofile(train_bin)
        val_ids.tofile(val_bin)
        
        print(f"Binary files saved to:")
        print(f"  - {train_bin}")
        print(f"  - {val_bin}")
        print("\nPreparation complete! You can now train your model with the chit-chat dataset.")
        
        # Print usage instructions
        print("\nTo train your model with this chit-chat dataset, run:")
        print("python train.py --dataset chitchat")
        
        # Create an example prompt file
        example_dir = os.path.join(os.path.dirname(data_dir), 'examples')
        if not os.path.exists(example_dir):
            os.makedirs(example_dir)
        
        example_file = os.path.join(example_dir, 'chitchat_prompt.txt')
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(f"{HUMAN_PREFIX}Hi\n\n{ASSISTANT_PREFIX}")
        
        print(f"\nAn example chit-chat prompt has been created at: {example_file}")
        print("Use this with gen.py to test your trained model:")
        print(f"python gen.py models/chitchat5000.pt --prompt_file examples/chitchat_prompt.txt")
        
    except Exception as e:
        print(f"Error saving binary files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
