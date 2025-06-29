# Jojo LLM Quick Start Guide

This guide provides quick examples for using the Jojo LLM project's text generation capabilities.

## Basic Text Generation

Generate text with default settings:

```bash
python gen.py models/story5000.pt
```

## Custom Prompt Generation

Generate text with a custom prompt:

```bash
python gen.py models/story5000.pt --prompt "Once upon a time in a digital forest, a small AI named Jojo was born."
```

## Nonstop Generation

Generate text continuously until manually stopped:

```bash
python gen.py models/story5000.pt --nonstop
```

## Interactive Mode

Enter interactive mode to generate multiple texts from different prompts:

```bash
python gen.py models/story5000.pt --interactive
```

## Creative Temperature Settings

Lower temperatures (e.g., 0.7) produce more focused and coherent text:

```bash
python gen.py models/story5000.pt --temp 0.7
```

Higher temperatures (e.g., 1.2) produce more diverse and creative text:

```bash
python gen.py models/story5000.pt --temp 1.2
```

## Saving Output

Save generated text to a file:

```bash
python gen.py models/story5000.pt --output stories/my_story.txt
```

In interactive mode, each generation will be saved with a timestamp:

```bash
python gen.py models/story5000.pt --interactive --output stories/session.txt
```

## Performance Optimization

For faster generation (no delay between tokens):

```bash
python gen.py models/story5000.pt --no_delay
```

Using specific precision:

```bash
python gen.py models/story5000.pt --dtype float16
```

## Different Devices

Run on CPU:

```bash
python gen.py models/story5000.pt --device cpu
```

Run on a specific CUDA device:

```bash
python gen.py models/story5000.pt --device cuda:1
```

## Loading Prompts from Files

Load a longer prompt from a text file:

```bash
python gen.py models/story5000.pt --prompt_file prompts/story_starter.txt
```
