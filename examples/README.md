# Jojo Examples

This directory contains examples and resources to help you get started with the Jojo LLM project.

## Contents

- **quick_start.md**: A quick reference guide with example commands for text generation
- **story_starter.txt**: A sample prompt that can be used with the `--prompt_file` option

## Using Example Files

To use the example story starter prompt:

```bash
python gen.py --prompt_file examples/story_starter.txt
```

## Creating Your Own Prompts

You can create your own prompt files in any text editor. Place them in this directory or any location of your choice, then use the `--prompt_file` option to specify the path when running the generator.

For best results:
- Start with a clear context or scenario
- Include some character or setting details
- End with an open-ended statement or dialogue to give the model direction

## Example Output Directory

When using the `--output` option, generated text will be saved to the specified file path. If you specify a path in a non-existent directory, the directory will be created automatically.

Example:
```bash
python gen.py --prompt_file examples/story_starter.txt --output output/my_story.txt
```
