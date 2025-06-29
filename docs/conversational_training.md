# Conversational Training Guide for Jojo LLM

This guide explains how to train Jojo with conversational datasets to improve its small talk abilities.

## Available Conversational Datasets

### DailyDialog
A high-quality multi-turn dialogue dataset that contains conversations focusing on everyday topics. It features:
- 13,118 human-written dialogues
- Clean, grammatical English text
- Multiple speakers/turns in each conversation
- Covers various daily conversation topics
- Perfect for training basic conversational abilities

## Setting Up DailyDialog

1. Run the preparation script to download and prepare the dataset:
   ```bash
   python data/prepare-dailydialog.py
   ```

2. Train your model on the DailyDialog dataset:
   ```bash
   python train.py --dataset dailydialog
   ```

3. For longer training (recommended for better results):
   ```bash
   python train.py --dataset dailydialog --max_iters 10000
   ```

## Testing Conversational Skills

After training, test the model's conversational abilities:

```bash
python gen.py models/dailydialog5000.pt --interactive
```

Try prompts like:
- "Person A: Hello, how are you today?"
- "Person A: What do you like to do on weekends?"
- "Person A: I just watched an amazing movie."

## Tips for Better Conversational Results

1. **Training Duration**: Conversation requires more training than basic text generation. Consider increasing `--max_iters` to 10000 or higher for better results.

2. **Prompt Formatting**: When testing, format your prompts as they appear in the training data:
   ```
   Person A: [Your message here]
   ```

3. **Temperature Settings**: For more natural conversation, use a slightly higher temperature:
   ```bash
   python gen.py models/dailydialog5000.pt --temp 0.8 --interactive
   ```

## Other Good Conversational Datasets

If you want to expand Jojo's conversational abilities further, consider these datasets:

1. **ConvAI2 (Persona-Chat)** - Conversations with assigned personas
2. **BlendedSkillTalk** - Combines empathy, knowledge, and personality
3. **Cornell Movie-Dialogs Corpus** - Movie script conversations
4. **The Empathetic Dialogues dataset** - Emotionally aware conversations

Each dataset would require its own preparation script similar to `prepare-dailydialog.py`.
