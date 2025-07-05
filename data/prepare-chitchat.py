#!/usr/bin/python3
"""
Jojo LLM Data Preparation - Simple Chit-Chat Dataset

This script creates a simple chit-chat dataset with common greetings
and short conversational exchanges to help train the model for 
basic social interactions. The dataset includes a diverse set of
user names and conversational templates to help the model generalize
better across different users and interaction styles.

Key features:
- Uses ChatML-style special tokens for formatting
- Includes diverse names across different cultures
- Generates varied greeting, question, and multi-turn conversations
- Supports name introduction and personalized responses
- Includes nickname handling and personalization

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
import json

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
SYSTEM_PROMPTS = [
    "You are a helpful, harmless, and honest AI assistant.",
    "You are an AI assistant designed to assist users with their questions and tasks.",
    "You are Jojo, an AI assistant created to help users with information and tasks.",
    "You are a friendly AI assistant ready to help with any questions or tasks.",
    "You are an AI assistant trained to provide accurate and helpful responses."
]

# Helper functions for new format
def format_user_message(message):
    """Format a message from the user with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['user']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_assistant_message(message):
    """Format a message from the assistant with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_system_message(message):
    """Format a system message with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['system']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Generate a set of common greetings, questions, and responses
def generate_chitchat_dataset():
    """
    Generate a simple dataset of common chit-chat exchanges.
    Each exchange consists of a system prompt, human greeting/question, and assistant response.
    Returns a list of conversations (each as a list of ChatML-formatted turns).
    """
    chitchat_data = []
    
    # List of diverse names for personalization
    diverse_names = [
        # English/Western names
        "Jason", "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", 
        "James", "Isabella", "Oliver", "Charlotte", "Benjamin", "Amelia", 
        "Elijah", "Mia", "Lucas", "Harper", "Mason", "Evelyn", "Logan", "Abigail",
        "Alexander", "Emily", "Ethan", "Elizabeth", "Jacob", "Sofia", "Michael", "Avery",
        "Daniel", "Ella", "Henry", "Scarlett", "Jackson", "Grace", "Sebastian", "Chloe",
        "Matthew", "Victoria", "David", "Riley", "Joseph", "Aria", "Carter", "Lily",
        "Owen", "Aubrey", "Wyatt", "Zoey", "John", "Penelope", "Jack", "Layla", 
        "Luke", "Lillian", "Jayden", "Nora", "Dylan", "Camila", "Gabriel", "Hannah",
        "Jane", "Jonathan", "Julia", "Jessica", "Jenna", "Jordan", "Jasmine", "Jared",

        # European names
        "Louis", "Chloe", "Gabriel", "Camille", "Arthur", "Léa", "Raphaël", "Manon",
        "Jules", "Inès", "Lucas", "Lola", "Nathan", "Clara", "Adam", "Zoé",
        "Maximilian", "Sophie", "Alexander", "Marie", "Paul", "Emilia", "Leon", "Hannah",
        "Felix", "Mia", "Jonas", "Lukas", "Anna", "Tim", "Lea",
        "Emil", "Freja", "Oscar", "Ida", "Oliver", "Astrid", "Hugo", "Ella",
        "Anton", "Signe", "Mikkel", "Mathilde", "Victor", "William", "Sofie", "Amalie",
        "Alma", "Sebastian", "Maja", "Lars", "Noah", "Nora",
        
        # South Asian names
        "Aarav", "Aanya", "Vivaan", "Anaya", "Vihaan", "Saanvi", "Arjun", "Anika",
        "Reyansh", "Ishani", "Ayaan", "Pari", "Atharv", "Anvi", "Shaurya", "Myra",
        "Advik", "Aarohi", "Rudra", "Kiara", "Kabir", "Navya", "Aditya", "Ishita",
        "Krish", "Riya", "Aryan", "Shreya", "Arnav", "Diya", "Nikhil", "Aditi",
        "Raj", "Priya", "Sanjay", "Divya", "Rahul", "Neha", "Vijay", "Pooja",
        
        # East Asian names
        "Wei", "Mei", "Hiroshi", "Yuki", "Chen", "Jing", "Kenji", "Sakura",
        "Takashi", "Ayumi", "Ming", "Lin", "Ryo", "Yuna", "Zheng", "Xiu",
        "Taro", "Hana", "Jin", "Eun", "Hikaru", "Akiko", "Akira", "Emi",
        "Li", "Ying", "Takumi", "Rin", "Jun", "Natsuki", "Kenta", "Yue",
        "Hayato", "Aoi", "Sora", "Kaori", "Haruto", "Yumi", "Kei", "Mizuki",

        # Middle Eastern/Arabic names
        "Amir", "Fatima", "Omar", "Zara", "Ali", "Leila", "Hassan", "Jasmine", 
        "Mohammed", "Layla", "Ibrahim", "Amira", "Yusuf", "Noor", "Ahmad", "Rania",
        "Zaid", "Samira", "Tariq", "Dalia", "Malik", "Zahra", "Karim", "Hana",
        "Mustafa", "Lina", "Samir", "Maya", "Jamal", "Nadia", "Fadi", "Mariam",
        
        # Hispanic/Latino names
        "Miguel", "Sofia", "Diego", "Isabella", "Mateo", "Valentina", "Santiago", "Camila",
        "Sebastian", "Emma", "Alejandro", "Lucia", "Leonardo", "Victoria", "Gabriel", "Valeria",
        "Emiliano", "Mariana", "Daniel", "Daniela", "Adrian", "Regina", "David", "Elena",
        "Jose", "Gabriela", "Luis", "Natalia", "Javier", "Ximena", "Carlos", "Paula",
        
        # African names
        "Amara", "Kwame", "Nia", "Kofi", "Zuri", "Sekou", "Ayanna", "Tendai", 
        "Folami", "Jelani", "Makena", "Chike", "Aaliyah", "Tafari", "Imani", "Faraji",
        "Adhiambo", "Bakari", "Thema", "Jabari", "Kesi", "Olu", "Zalika", "Kwesi",
        "Abeni", "Nnamdi", "Fola", "Oluchi", "Chiamaka", "Dakarai", "Zola", "Kamau"
    ]
    
    # Generate name introduction interactions with multiple names
    name_interactions = []
    
    # Templates for name introductions
    name_intro_templates = [
        "My name is {}.",
        "I am {}.",
        "You can call me {}.",
        "Hi, I'm {}.",
        "Hello, my name is {}.",
        "I'd like to introduce myself. I'm {}.",
        "Please address me as {}.",
        "{}. That's my name.",
        "I go by {}.",
        "I prefer to be called {}.",
        "Everyone calls me {}.",
        "Nice to meet you, I'm {}.",
        "The name's {}.",
        "Allow me to introduce myself. I'm {}.",
        "{}. But my friends call me {}."  # This one will be handled specially
    ]
    
    # Templates for assistant responses with {name} placeholder
    assistant_response_templates = [
        "Hi {name}, it's nice to meet you! How may I help you today?",
        "Hello {name}! What can I assist you with today?",
        "Nice to meet you, {name}! How can I help you?",
        "Hi {name}! I'm Jojo. What can I do for you today?",
        "Hello {name}! It's great to meet you. What can I help you with today?",
        "Nice to meet you, {name}. I'm Jojo. How may I assist you?",
        "I'll call you {name}! It's nice to meet you. How can I assist you today?",
        "Hello {name}! I'm Jojo. What can I help you with?",
        "It's a pleasure to meet you, {name}. How can I be of service today?",
        "Welcome, {name}! Is there anything specific you'd like help with?",
        "Great to meet you, {name}! How can I make your day better?",
        "Hello {name}! I'm Jojo, an AI assistant. What brings you here today?",
        "Nice to meet you, {name}! I'm here to help with whatever you need.",
        "Hi {name}! Thanks for introducing yourself. How can I assist you?",
        "Hello {name}! I'm Jojo. I'll remember your name during our conversation. How can I help you today?"
    ]
    
    # Function to generate nickname from full name
    def generate_nickname(name):
        if len(name) <= 3:
            return name  # Name is too short for a nickname
        
        # Common nickname patterns
        if name.lower().startswith('ch'):
            return name[:3]  # "Charles" -> "Cha"
        elif name.lower().startswith('j'):
            return name[:4] if len(name) >= 4 else name  # "James" -> "Jame"
        elif name.lower().endswith('y'):
            return name  # Already sounds like a nickname
        elif name.lower().endswith('ia'):
            return name[:-2] + "y"  # "Victoria" -> "Victory"
        elif len(name) >= 5:
            return name[:3]  # First 3 letters
        else:
            return name
    
    # Generate combinations of names with introductions and responses
    for name in diverse_names:
        # Sample 3 random templates per name
        for intro_template in random.sample(name_intro_templates, 3):
            # Handle the special case with nickname
            if "{}. But my friends call me {}" in intro_template:
                nickname = generate_nickname(name)
                if nickname == name:  # If nickname generation failed, use a simple truncation
                    nickname = name[:3] if len(name) > 3 else name
                    
                human_intro = intro_template.format(name, nickname)
                
                # Generate responses that acknowledge both name and nickname
                assistant_responses = []
                for _ in range(3):
                    response = random.choice([
                        f"Nice to meet you, {name}! I'll remember that you prefer {nickname}. How can I help you today?",
                        f"Hello {name}! I'll call you {nickname} as you prefer. What can I assist you with?",
                        f"Great to meet you, {nickname}! Thanks for letting me know your name. How can I help?",
                        f"Hi {nickname}! It's a pleasure to meet you. What brings you here today?"
                    ])
                    assistant_responses.append(response)
                
                name_interactions.append((human_intro, assistant_responses))
            else:
                human_intro = intro_template.format(name)
                
                # Generate 3 varied responses for each introduction
                assistant_responses = []
                for template in random.sample(assistant_response_templates, 3):
                    assistant_responses.append(template.format(name=name))
                    
                name_interactions.append((human_intro, assistant_responses))
            
    # Print some statistics about the generated data
    print(f"Generated {len(name_interactions)} name introduction interactions with {len(diverse_names)} different names")
    
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
    print("Generating greeting exchanges...", end="")
    for greeting, responses in greetings:
        for response in responses:
            conversation = [
                format_user_message(greeting),
                format_assistant_message(response)
            ]
            chitchat_data.append(conversation)
    print(f"  ({len(chitchat_data)} total exchanges)")
    
    # Generate single-turn conversations from questions
    print("Generating question exchanges...", end="")
    for question, responses in questions:
        for response in responses:
            conversation = [
                format_user_message(question),
                format_assistant_message(response)
            ]
            chitchat_data.append(conversation)
    print(f"  ({len(chitchat_data)} total exchanges)")

    # Generate two-turn conversations (greeting + follow-up)
    print("Generating multi-turn exchanges...",end="")
    for greeting, greeting_responses in greetings:
        for greeting_response in greeting_responses[:2]:  # Limit to avoid too many combinations
            for follow_up, follow_up_responses in follow_ups:
                for follow_up_response in follow_up_responses[:1]:  # Just one response per follow-up
                    conversation = [
                        format_user_message(greeting),
                        format_assistant_message(greeting_response),
                        format_user_message(follow_up),
                        format_assistant_message(follow_up_response)
                    ]
                    chitchat_data.append(conversation)
    print(f"  ({len(chitchat_data)} total exchanges)")
    
    # Generate name introduction conversations
    print("Generating name introduction exchanges...", end="")
    for name_intro, responses in name_interactions:
        for response in responses:
            conversation = [
                format_user_message(name_intro),
                format_assistant_message(response)
            ]
            chitchat_data.append(conversation)
    print(f"  ({len(chitchat_data)} total exchanges)")

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
    ]
    
    # Generate name-based complex conversations using the diverse_names list
    name_complex_conversations = []
    
    # Templates for name-based complex conversations
    name_conversation_templates = [
        # Template 0: Nickname conversation
        [
            "My name is {name}, but you can call me {nickname}.",
            "Nice to meet you, {nickname}! I'll remember that you prefer to be called {nickname}. How can I help you today?",
            "Do you remember my full name and my nickname?",
            "Yes, your full name is {name}, but you prefer to be called {nickname}. I'll make sure to address you as {nickname} during our conversation. Is there anything I can help you with?"
        ],
        # Template 1: What is your name?
        [
            "What is your name?",
            "My name is Jojo. How can I help you today?",
            "My name is {name}",
            "It's nice to meet you, {name}! How may I assist you today?"
        ],
        # Template 2: Introduction and remembering
        [
            "Hi, I'm {name}",
            "Hello {name}! It's nice to meet you. I'm Jojo, your AI assistant. How can I help you today?",
            "Can you remember my name?",
            "Yes, I'll remember that your name is {name} during our conversation. How can I assist you, {name}?"
        ],
        # Template 3: Longer introduction
        [
            "Hello there, I'd like to introduce myself. My name is {name}.",
            "Hi {name}! It's a pleasure to meet you. I'm Jojo, an AI assistant. How can I help you today?",
            "Thanks for the warm welcome, Jojo. I'm just exploring what you can do.",
            "I'm happy to help you explore, {name}! I can answer questions, provide information, assist with various tasks, and engage in conversation. Is there anything specific you'd like to know?"
        ],
        # Template 4: Formal introduction
        [
            "Good day. My name is {name} and I'm new here.",
            "Welcome, {name}! I'm Jojo, your AI assistant. It's great to have you here. How can I make your experience better today?",
            "Could you tell me what capabilities you have?",
            "Of course, {name}. I can answer questions, provide information on various topics, help with writing and creative tasks, engage in conversation, and much more. What would you like to explore first?"
        ],
        # Template 5: Personal preference question
        [
            "Hello Jojo, I'm {name}.",
            "Hello {name}! It's nice to meet you. How can I help you today?",
            "What are some of your favorite things?",
            "As an AI assistant, I don't have personal preferences in the same way humans do. I'm designed to be helpful, informative, and engaging for users like you, {name}. Is there anything specific you'd like to know or discuss?"
        ],
        # Template 6: Technical help
        [
            "Hi Jojo, my name is {name} and I need some technical help.",
            "Hello {name}! I'd be happy to help with any technical issues you're facing. What seems to be the problem?",
            "I'm trying to figure out how to use this new app.",
            "I understand that can be challenging, {name}. To help you better, could you tell me which app you're trying to use? Once I know that, I can provide more specific guidance on how to get started."
        ],
        # Template 7: Asking about hobbies
        [
            "Hi Jojo, I'm {name}. What do you like to do for fun?",
            "Hello {name}! As an AI assistant, I don't have personal experiences or hobbies like humans do. I'm here to help you with information, answer your questions, and assist with tasks. How can I help you today?",
            "I enjoy hiking and reading. Do you have any book recommendations?",
            "Those are wonderful hobbies, {name}! While I don't have personal reading experiences, I can suggest some popular books in various genres. What kind of books do you typically enjoy reading?"
        ],
        # Template 8: Weather conversation
        [
            "Hi, my name is {name}. How's the weather today?",
            "Hello {name}! As an AI, I don't have access to real-time weather information or your location. To get accurate weather updates, you could check a weather app, website, or look outside. Is there something else I can help you with?",
            "It's sunny here. Just wanted to chat.",
            "That sounds lovely, {name}! Sunny days can be quite energizing. I'm always happy to chat. Is there any particular topic you'd like to discuss today?"
        ],
        # Template 9: Seeking advice
        [
            "Hello, I'm {name} and I'm looking for some advice.",
            "Hi {name}! I'd be happy to help with advice. What kind of guidance are you looking for today?",
            "I'm trying to be more productive. Any suggestions?",
            "That's a great goal, {name}! For improved productivity, you might try techniques like time blocking, the Pomodoro method (25 minutes of focused work followed by 5-minute breaks), minimizing distractions, prioritizing tasks with a to-do list, and ensuring you get adequate rest. Would you like me to elaborate on any of these methods?"
        ],
        # Template 10: Learning about user
        [
            "Hi Jojo! My name is {name}.",
            "Hello {name}! It's nice to meet you. How can I assist you today?",
            "I'm interested in learning about artificial intelligence.",
            "That's a fascinating field, {name}! Artificial Intelligence covers many areas like machine learning, natural language processing, computer vision, and robotics. Are you interested in a specific aspect of AI, or would you like a general introduction to the field?"
        ]
    ]
    
    # Generate conversations with different names
    for template_idx, template in enumerate(name_conversation_templates):
        for name in random.sample(diverse_names, 5):  # Use 5 different names for each template
            conversation = []
            
            # Handle the nickname template specially
            if template_idx == 0:  # Nickname conversation template
                nickname = generate_nickname(name)
                if nickname == name:  # If nickname generation failed, use a simple variation
                    if len(name) > 3:
                        nickname = name[:3]
                    else:
                        nickname = name + "y"  # Add 'y' for very short names
                
                for i, msg in enumerate(template):
                    conversation.append(msg.format(name=name, nickname=nickname))
            else:
                # Regular templates without nicknames
                for i, msg in enumerate(template):
                    conversation.append(msg.format(name=name))
                    
            name_complex_conversations.append(conversation)
    
    # Add the name-based complex conversations to the main list
    complex_conversations.extend(name_complex_conversations)
    
    print(f"Generated {len(name_complex_conversations)} complex conversations with diverse names")
    
    # Instead of joining with \n, keep each conversation as a list of turns (strings)
    return chitchat_data

# New function to wrap conversations with a system prompt and output as JSONL

def write_jsonl_conversations(conversations, output_path, system_prompts=SYSTEM_PROMPTS):
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            system_prompt = random.choice(system_prompts)
            chatml = [f"<|im_start|>system\n{system_prompt}\n<|im_end|>"] + conv
            chatml.append("<|endoftext|>")
            text = "\n".join(chatml)
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

def main():
    print("\n=== Jojo LLM Chit-Chat Dataset Preparation ===\n")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(data_dir, 'chitchat')
    ensure_dir(output_base)
    # Generate the dataset
    chitchat_data = generate_chitchat_dataset()
    # Shuffle and split into train/val
    random.shuffle(chitchat_data)
    split = int(0.95 * len(chitchat_data))
    train_convs = chitchat_data[:split]
    val_convs = chitchat_data[split:]
    # Write JSONL files
    write_jsonl_conversations(train_convs, os.path.join(data_dir, 'chitchat-train.jsonl'))
    write_jsonl_conversations(val_convs, os.path.join(data_dir, 'chitchat-val.jsonl'))
    print(f"Wrote {len(train_convs)} train and {len(val_convs)} val conversations to JSONL files.")

if __name__ == "__main__":
    main()
