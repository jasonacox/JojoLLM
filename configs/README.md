# Configuraiton

This directory contains example configuration files that can be used with the Jojo Trainier

## Hyperparameters

| Model | Layers | Heads | Embedding Dimension | Parameters |
|-------|--------|-------|---------------------|------------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

## Running

```bash
# Use a configuration file
python train.py --config configs/gpt2-medium.json
```
