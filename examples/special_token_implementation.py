#!/usr/bin/python3
"""
Jojo LLM - Special Tokens Implementation Example

This is an example prototype showing how to modify the project
to use special tokens for chat roles instead of plain text markers.

Author: GitHub Copilot
2025 June 28
"""

import os
import tiktoken
import numpy as np
from tqdm import tqdm

# Define the special tokens we want to use
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "endoftext": "<|endoftext|>"
}

# Current format constants
HUMAN_PREFIX = "Human: "
ASSISTANT_PREFIX = "Assistant: "
TURN_SEPARATOR = "\n\n"
CONVERSATION_SEPARATOR = "\n\n<|endoftext|>\n\n"

# New format constants
def format_user_message(message):
    """Format a message from the user with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['user']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_assistant_message(message):
    """Format a message from the assistant with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

def format_system_message(message):
    """Format a system message with special tokens"""
    return f"{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['system']}\n{message.strip()}\n{SPECIAL_TOKENS['im_end']}"

# Example function to convert existing conversations to the new format
def convert_conversation(conversation):
    """
    Convert a conversation from the old format to the new format
    
    Args:
        conversation (str): A conversation in the old format
            e.g. "Human: Hello\\n\\nAssistant: Hi there"
            
    Returns:
        str: The conversation in the new format
            e.g. "<|im_start|>user\\nHello\\n<|im_end|>\\n<|im_start|>assistant\\nHi there\\n<|im_end|>"
    """
    new_messages = []
    turns = conversation.split(TURN_SEPARATOR)
    
    for turn in turns:
        if turn.startswith(HUMAN_PREFIX):
            # Convert Human: message to user format
            message = turn[len(HUMAN_PREFIX):]
            new_messages.append(format_user_message(message))
        elif turn.startswith(ASSISTANT_PREFIX):
            # Convert Assistant: message to assistant format
            message = turn[len(ASSISTANT_PREFIX):]
            new_messages.append(format_assistant_message(message))
    
    # Join with a newline separator
    return "\n".join(new_messages)

# Example of how to modify the tokenizer to handle special tokens
def setup_tokenizer():
    """
    Setup the tokenizer with special tokens
    
    Note: This is a conceptual example. Actual implementation would depend on
    how you want to approach the tokenizer modification challenge.
    """
    # Here are several potential approaches:
    
    # Approach 1: Use tiktoken directly but handle special tokens separately
    enc = tiktoken.get_encoding("gpt2")
    
    # Approach 2 (conceptual): Add special tokens to a custom encoding
    # This would require modifying tiktoken or creating a custom tokenizer
    # custom_enc = create_custom_tokenizer(base="gpt2", special_tokens=SPECIAL_TOKENS)
    
    # Approach 3: Use a Hugging Face tokenizer instead
    # from transformers import GPT2Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})
    
    return enc

# Example of how to modify the data preparation in prepare-chat.py
def example_format_chat_template(dialogues):
    """
    Example of how to modify the format_chat_template function in prepare-chat.py
    to use the new special token format
    """
    formatted_data = []
    
    for dialogue in tqdm(dialogues, desc="Formatting dialogues", ncols=80):
        turns = dialogue.split('__eou__')
        # Remove empty turns and the trailing empty string after the last __eou__
        turns = [turn.strip() for turn in turns if turn.strip()]
        
        # Format as a conversation with human-assistant alternating turns
        if len(turns) >= 2:  # Only use dialogues with at least 2 turns
            conversation = []
            for i, turn in enumerate(turns):
                # Add a system message at the beginning (optional)
                if i == 0:
                    # You could add a system message here if desired
                    pass
                
                # Process each turn
                if i % 2 == 0:
                    # User turn
                    conversation.append(format_user_message(turn))
                else:
                    # Assistant turn
                    conversation.append(format_assistant_message(turn))
            
            # Join turns with newlines
            formatted_conversation = "\n".join(conversation)
            formatted_data.append(formatted_conversation)
    
    return formatted_data

# Example of how to modify the generation code in gen.py
def example_prepare_prompt_for_generation(prompt, add_assistant=True):
    """
    Example of how to modify prompt preparation in gen.py
    to use the new special token format
    """
    # Check if prompt already has special tokens
    if SPECIAL_TOKENS['im_start'] in prompt or SPECIAL_TOKENS['im_end'] in prompt:
        # Already in the new format - make sure it ends properly for generation
        if add_assistant and not prompt.endswith(SPECIAL_TOKENS['im_end']):
            if SPECIAL_TOKENS['im_start'] + SPECIAL_TOKENS['assistant'] in prompt:
                # Assistant already started but not ended, do nothing
                return prompt
            else:
                # Add assistant token for the model to continue as assistant
                return prompt.rstrip() + f"\n{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n"
        else:
            return prompt
    
    # Convert from old format if it contains Human: or Assistant:
    elif HUMAN_PREFIX in prompt or ASSISTANT_PREFIX in prompt:
        turns = []
        for line in prompt.split(TURN_SEPARATOR):
            if line.startswith(HUMAN_PREFIX):
                message = line[len(HUMAN_PREFIX):]
                turns.append(format_user_message(message))
            elif line.startswith(ASSISTANT_PREFIX):
                message = line[len(ASSISTANT_PREFIX):]
                turns.append(format_assistant_message(message))
        
        formatted = "\n".join(turns)
        
        # Add assistant start if needed
        if add_assistant:
            formatted += f"\n{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n"
            
        return formatted
    
    # If just plain text, assume it's a user message
    else:
        formatted = format_user_message(prompt)
        if add_assistant:
            formatted += f"\n{SPECIAL_TOKENS['im_start']}{SPECIAL_TOKENS['assistant']}\n"
        return formatted

# Example usage and testing
def main():
    # Example of conversation conversion
    old_conversation = """Human: Hello! How are you today?
Assistant: I'm doing great, thank you! How can I assist you today?
Human: I have a question about my order.
Assistant: Sure, I'd be happy to help with your order. Can you please provide your order number?
Human: It's 12345.
Assistant: Thank you for providing your order number. Let me check the details for you.
Assistant: I see that your order was delivered successfully. Is there anything else I can help you with?
Human: No, that's all. Thank you!
Assistant: You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!"""

    print("Old conversation format:")
    print(old_conversation)
    
    # Convert the old conversation to the new format
    new_conversation = convert_conversation(old_conversation)
    print("\nConverted to new conversation format:")
    print(new_conversation)
    
    # Setup the tokenizer (example, not used further here)
    tokenizer = setup_tokenizer()
    
    # Example dialogue data
    dialogues = [
        "Hello, how are you?__eou__I'm fine, thank you!__eou__What can I do for you today?__eou__",
        "Hi there!__eou__Hello! How can I help you?__eou__"
    ]
    
    # Format the chat template
    formatted_dialogues = example_format_chat_template(dialogues)
    print("\nFormatted dialogues:")
    for fd in formatted_dialogues:
        print(fd)
        print("---")
    
    # Example prompt preparation
    prompt = "Human: Tell me a joke.\n\nAssistant: Why did the chicken cross the road? To get to the other side!"
    prepared_prompt = example_prepare_prompt_for_generation(prompt)
    print("\nPrepared prompt for generation:")
    print(prepared_prompt)
    print("---")
    
    # Test the conversion of a single message
    test_message = "Human: What is the capital of France?"
    converted_message = convert_conversation(test_message)
    print("\nConverted single message:")
    print(converted_message)
    print("---")
    
    # Test formatting a user message
    user_message = "What is the weather like today?"
    formatted_user = format_user_message(user_message)
    print("\nFormatted user message:")
    print(formatted_user)
    print("---")
    
    # Test formatting an assistant message
    assistant_message = "It's sunny and warm today."
    formatted_assistant = format_assistant_message(assistant_message)
    print("\nFormatted assistant message:")
    print(formatted_assistant)
    print("---")
    
    # Test formatting a system message
    system_message = "You are a helpful assistant."
    formatted_system = format_system_message(system_message)
    print("\nFormatted system message:")
    print(formatted_system)
    print("---")

if __name__ == "__main__":
    main()
