# Chat Template Format for Jojo LLM

This document outlines the standardized chat format used for training Jojo LLM on conversational data and for interacting with the model once trained.

## Format Specification

The chat template uses the following format conventions:

1. **Role Prefixes**:
   - User messages start with: `Human: `
   - Model responses start with: `Assistant: `

2. **Turn Separator**:
   - Each turn (Human or Assistant) is separated by two newlines (`\n\n`)

3. **Conversation Separator**:
   - Multiple conversations in training data are separated by: `\n\n<|endoftext|>\n\n`

## Training Data Structure

For the training data, each conversation follows this pattern:

```
Human: [first user message]
Assistant: [first assistant response]
```

For multi-turn conversations, the alternating pattern continues:

```
Human: [first user message]
Assistant: [first assistant response]
Human: [second user message]
Assistant: [second assistant response]
...
```
