import sys
import argparse

parser = argparse.ArgumentParser(description="Convert ChatML-style file to training format.")
parser.add_argument('input', help='Input file (ChatML format)')
parser.add_argument('output', help='Output file (train format)')
args = parser.parse_args()

user_tag = '<|user|>'
assistant_tag = '<|assistant|>'
end_tag = '<|endoftext|>'

with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
    user_prompt = None
    assistant_prompt = None
    for line in fin:
        line = line.strip()
        if not line:
            continue
        if line.startswith(user_tag):
            user_prompt = line[len(user_tag):].strip()
        elif line.startswith(assistant_tag):
            assistant_prompt = line[len(assistant_tag):].strip()
        if user_prompt is not None and assistant_prompt is not None:
            fout.write(f"$user\n{user_prompt}\n")
            fout.write(f"$assistant\n{assistant_prompt}\n")
            fout.write(f"{end_tag}\n")
            user_prompt = None
            assistant_prompt = None
