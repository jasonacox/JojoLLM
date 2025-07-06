#!/usr/bin/env python3
"""
Hugging Face Model Upload Script for Jojo LLM

This script helps prepare and upload trained Jojo models to Hugging Face Hub.
It handles checkpoint conversion, model card creation, and the upload process.

Features:
- Convert Jojo checkpoints to Hugging Face format
- Generate comprehensive model cards
- Upload models with proper metadata
- Support for both public and private repositories
- Automatic tokenizer configuration
- Training metrics integration

Author: Jason A. Cox
2025 July 5
https://github.com/jasonacox/jojo
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from dataclasses import asdict

# Try to import Hugging Face libraries
try:
    from transformers import (
        GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
        AutoTokenizer, AutoModelForCausalLM
    )
    from huggingface_hub import HfApi, create_repo, upload_folder
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Hugging Face libraries not found. Install with: pip install transformers huggingface_hub")

# Import Jojo modules
try:
    from model import GPT, GPTConfig
    from config import Config, Constants
    from setup_tokenizer import get_extended_tokenizer
    JOJO_MODULES_AVAILABLE = True
except ImportError:
    JOJO_MODULES_AVAILABLE = False

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=f'{BLUE}%(asctime)s{ENDC} {GREEN}%(levelname)s:{ENDC} %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

class JojoToHuggingFaceConverter:
    """Convert Jojo models to Hugging Face format"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def load_jojo_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load Jojo checkpoint and extract information"""
        self.logger.info(f"Loading Jojo checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model configuration
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if isinstance(config_dict, dict):
                model_config = config_dict.get('model', {})
            else:
                model_config = {}
        else:
            # Fallback: try to extract from model_args
            model_args = checkpoint.get('model_args', {})
            model_config = {
                'n_layer': model_args.get('n_layer', 12),
                'n_head': model_args.get('n_head', 12),
                'n_embd': model_args.get('n_embd', 768),
                'block_size': model_args.get('block_size', 1024),
                'dropout': model_args.get('dropout', 0.0),
                'bias': model_args.get('bias', False),
                'vocab_size': model_args.get('vocab_size', 50304)
            }
        
        self.logger.info(f"Model configuration: {model_config}")
        
        return {
            'state_dict': checkpoint['model'],
            'model_config': model_config,
            'training_config': config_dict if 'config' in checkpoint else {},
            'metadata': checkpoint.get('metadata', {}),
            'epoch': checkpoint.get('epoch', 0),
            'batch_counter': checkpoint.get('batch_counter', 0),
            'best_val_loss': checkpoint.get('best_val_loss', None),
            'metrics': checkpoint.get('metrics', {})
        }
    
    def convert_to_huggingface_config(self, model_config: Dict[str, Any]) -> GPT2Config:
        """Convert Jojo model config to Hugging Face GPT2Config"""
        
        # Map Jojo config to HF config
        hf_config = GPT2Config(
            vocab_size=model_config.get('vocab_size', 50304),
            n_positions=model_config.get('block_size', 1024),
            n_embd=model_config.get('n_embd', 768),
            n_layer=model_config.get('n_layer', 12),
            n_head=model_config.get('n_head', 12),
            activation_function="gelu_new",
            resid_pdrop=model_config.get('dropout', 0.0),
            embd_pdrop=model_config.get('dropout', 0.0),
            attn_pdrop=model_config.get('dropout', 0.0),
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=model_config.get('dropout', 0.0),
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            return_dict=True,
        )
        
        self.logger.info(f"Created Hugging Face config: {hf_config}")
        return hf_config
    
    def convert_state_dict(self, jojo_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert Jojo state dict to Hugging Face format"""
        
        hf_state_dict = {}
        
        # Mapping from Jojo parameter names to HF parameter names
        name_mapping = {
            # Token embeddings
            'transformer.wte.weight': 'transformer.wte.weight',
            'transformer.wpe.weight': 'transformer.wpe.weight',
            
            # Layer norm
            'transformer.ln_f.weight': 'transformer.ln_f.weight',
            'transformer.ln_f.bias': 'transformer.ln_f.bias',
            
            # LM head
            'lm_head.weight': 'lm_head.weight',
        }
        
        # Handle transformer blocks
        for key, value in jojo_state_dict.items():
            if key.startswith('transformer.h.'):
                # Extract layer number and parameter name
                parts = key.split('.')
                layer_num = parts[2]  # transformer.h.{layer_num}.{param}
                param_path = '.'.join(parts[3:])  # {param}
                
                # Map parameter names within each layer
                layer_mapping = {
                    'ln_1.weight': f'transformer.h.{layer_num}.ln_1.weight',
                    'ln_1.bias': f'transformer.h.{layer_num}.ln_1.bias',
                    'attn.c_attn.weight': f'transformer.h.{layer_num}.attn.c_attn.weight',
                    'attn.c_attn.bias': f'transformer.h.{layer_num}.attn.c_attn.bias',
                    'attn.c_proj.weight': f'transformer.h.{layer_num}.attn.c_proj.weight',
                    'attn.c_proj.bias': f'transformer.h.{layer_num}.attn.c_proj.bias',
                    'ln_2.weight': f'transformer.h.{layer_num}.ln_2.weight',
                    'ln_2.bias': f'transformer.h.{layer_num}.ln_2.bias',
                    'mlp.c_fc.weight': f'transformer.h.{layer_num}.mlp.c_fc.weight',
                    'mlp.c_fc.bias': f'transformer.h.{layer_num}.mlp.c_fc.bias',
                    'mlp.c_proj.weight': f'transformer.h.{layer_num}.mlp.c_proj.weight',
                    'mlp.c_proj.bias': f'transformer.h.{layer_num}.mlp.c_proj.bias',
                }
                
                if param_path in layer_mapping:
                    hf_state_dict[layer_mapping[param_path]] = value
                else:
                    # Fallback: use original name
                    hf_state_dict[key] = value
            elif key in name_mapping:
                hf_state_dict[name_mapping[key]] = value
            else:
                # For any unmapped parameters, use original name
                hf_state_dict[key] = value
        
        self.logger.info(f"Converted {len(jojo_state_dict)} parameters to {len(hf_state_dict)} HF parameters")
        return hf_state_dict
    
    def convert_model(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        """Convert complete Jojo model to Hugging Face format"""
        
        # Load Jojo checkpoint
        checkpoint_data = self.load_jojo_checkpoint(checkpoint_path)
        
        # Create HF config
        hf_config = self.convert_to_huggingface_config(checkpoint_data['model_config'])
        
        # Convert state dict
        hf_state_dict = self.convert_state_dict(checkpoint_data['state_dict'])
        
        # Create HF model
        hf_model = GPT2LMHeadModel(hf_config)
        
        # Load converted weights
        try:
            hf_model.load_state_dict(hf_state_dict, strict=False)
            self.logger.info("Successfully loaded converted weights into HF model")
        except Exception as e:
            self.logger.warning(f"Error loading weights (non-critical): {e}")
            # Try loading with strict=False
            missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
            if missing_keys:
                self.logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and config
        hf_model.save_pretrained(output_dir)
        self.logger.info(f"Saved HF model to {output_dir}")
        
        return {
            'model_config': checkpoint_data['model_config'],
            'training_config': checkpoint_data['training_config'],
            'metadata': checkpoint_data['metadata'],
            'metrics': checkpoint_data['metrics'],
            'hf_config': hf_config
        }

class ModelCardGenerator:
    """Generate comprehensive model cards for Hugging Face"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def generate_model_card(self, 
                          model_info: Dict[str, Any],
                          model_name: str,
                          dataset_name: str,
                          checkpoint_path: str,
                          additional_info: Dict[str, Any] = None) -> str:
        """Generate a comprehensive model card"""
        
        additional_info = additional_info or {}
        model_config = model_info.get('model_config', {})
        training_config = model_info.get('training_config', {})
        metadata = model_info.get('metadata', {})
        metrics = model_info.get('metrics', {})
        
        # Extract key metrics
        best_val_loss = None
        final_train_loss = None
        if metrics:
            # Try to get the best validation loss
            val_losses = metrics.get('val_loss', [])
            if val_losses:
                best_val_loss = min([loss for _, loss in val_losses])
            
            # Try to get final training loss
            train_losses = metrics.get('train_loss', [])
            if train_losses:
                final_train_loss = train_losses[-1][1] if train_losses else None
        
        # Calculate model size
        total_params = 0
        if model_config:
            n_layer = model_config.get('n_layer', 12)
            n_head = model_config.get('n_head', 12)
            n_embd = model_config.get('n_embd', 768)
            vocab_size = model_config.get('vocab_size', 50304)
            
            # Rough calculation of parameters
            total_params = (
                vocab_size * n_embd +  # Token embeddings
                model_config.get('block_size', 1024) * n_embd +  # Position embeddings
                n_layer * (
                    4 * n_embd * n_embd +  # Attention weights
                    4 * n_embd * 4 * n_embd +  # MLP weights
                    2 * n_embd  # Layer norms
                ) +
                n_embd +  # Final layer norm
                vocab_size * n_embd  # LM head
            )
        
        model_card = f"""---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt
- language-model
- jojo-llm
- pytorch
datasets:
- {dataset_name}
metrics:
- perplexity
model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: {dataset_name}
      name: {dataset_name.title()}
    metrics:
    - type: perplexity
      value: {f"{2**best_val_loss:.2f}" if best_val_loss else "N/A"}
      name: Perplexity
---

# {model_name}

## Model Description

{model_name} is a GPT-style language model trained using the Jojo LLM training framework. This model was fine-tuned on the {dataset_name} dataset and is designed for text generation tasks.

## Model Details

- **Model Type**: GPT-style Transformer Language Model
- **Training Framework**: [Jojo LLM](https://github.com/jasonacox/jojo)
- **Language**: English
- **License**: MIT

### Architecture

- **Layers**: {model_config.get('n_layer', 'N/A')}
- **Hidden Size**: {model_config.get('n_embd', 'N/A')}
- **Attention Heads**: {model_config.get('n_head', 'N/A')}
- **Context Length**: {model_config.get('block_size', 'N/A')} tokens
- **Vocabulary Size**: {model_config.get('vocab_size', 'N/A'):,}
- **Total Parameters**: {f"{total_params/1e6:.1f}M" if total_params else "N/A"}

## Training Details

### Training Data

The model was trained on the **{dataset_name}** dataset. 

### Training Procedure

- **Training Framework**: Jojo LLM v{metadata.get('trainer_version', Constants.VERSION)}
- **PyTorch Version**: {metadata.get('pytorch_version', 'N/A')}
- **Training Device**: {training_config.get('system', {}).get('device', 'N/A')}
- **Precision**: {training_config.get('system', {}).get('dtype', 'N/A')}

#### Training Hyperparameters

- **Batch Size**: {training_config.get('training', {}).get('batch_size', 'N/A')}
- **Gradient Accumulation Steps**: {training_config.get('training', {}).get('gradient_accumulation_steps', 'N/A')}
- **Learning Rate**: {training_config.get('optimizer', {}).get('learning_rate', 'N/A')}
- **Weight Decay**: {training_config.get('optimizer', {}).get('weight_decay', 'N/A')}
- **Dropout**: {model_config.get('dropout', 'N/A')}
- **Gradient Clipping**: {training_config.get('optimizer', {}).get('grad_clip', 'N/A')}

#### Training Results

{f"- **Final Training Loss**: {final_train_loss:.4f}" if final_train_loss else ""}
{f"- **Best Validation Loss**: {best_val_loss:.4f}" if best_val_loss else ""}
{f"- **Perplexity**: {2**best_val_loss:.2f}" if best_val_loss else ""}

## Usage

### Using with Transformers

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("{additional_info.get('repo_name', model_name)}")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text
input_text = "Your prompt here"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Using with Jojo LLM

```bash
# Generate text using the original Jojo framework
python gen.py {os.path.basename(checkpoint_path)} --prompt "Your prompt here"
```

## Technical Specifications

- **Model Format**: PyTorch
- **Precision**: {training_config.get('system', {}).get('dtype', 'bfloat16')}
- **Framework Compatibility**: 
  - ✅ Hugging Face Transformers
  - ✅ Jojo LLM
  - ✅ PyTorch

## Model Card Authors

This model card was automatically generated by the Jojo LLM Hugging Face upload script.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{model_name.lower().replace('-', '_')},
  title={{{model_name}}},
  author={{Jason A. Cox}},
  year={{2025}},
  howpublished={{\\url{{https://github.com/jasonacox/jojo}}}},
  note={{Trained using Jojo LLM framework}}
}}
```

## Framework Information

- **Jojo LLM Version**: {metadata.get('trainer_version', Constants.VERSION)}
- **Generation Date**: {metadata.get('timestamp', 'N/A')}
- **Checkpoint**: `{os.path.basename(checkpoint_path)}`

For more information about the Jojo LLM framework, visit: https://github.com/jasonacox/jojo
"""

        return model_card

class HuggingFaceUploader:
    """Handle uploading to Hugging Face Hub"""
    
    def __init__(self, logger):
        self.logger = logger
        self.api = HfApi()
    
    def setup_tokenizer(self, output_dir: str, tokenizer_type: str = "gpt2"):
        """Setup tokenizer for the model"""
        
        if tokenizer_type == "extended":
            try:
                # Try to use the extended tokenizer from Jojo
                tokenizer = get_extended_tokenizer()
                # Save as GPT2 tokenizer format
                tokenizer.save_pretrained(output_dir)
                self.logger.info("Saved extended Jojo tokenizer")
            except Exception as e:
                self.logger.warning(f"Could not save extended tokenizer: {e}")
                # Fallback to standard GPT2 tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.save_pretrained(output_dir)
                self.logger.info("Saved standard GPT2 tokenizer")
        else:
            # Use standard GPT2 tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(output_dir)
            self.logger.info("Saved standard GPT2 tokenizer")
    
    def create_repository(self, repo_name: str, private: bool = False, 
                         organization: Optional[str] = None) -> str:
        """Create repository on Hugging Face Hub"""
        
        full_repo_name = f"{organization}/{repo_name}" if organization else repo_name
        
        try:
            url = create_repo(
                repo_id=full_repo_name,
                private=private,
                repo_type="model"
            )
            self.logger.info(f"Created repository: {full_repo_name}")
            return full_repo_name
        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.info(f"Repository {full_repo_name} already exists")
                return full_repo_name
            else:
                raise e
    
    def upload_model(self, local_dir: str, repo_name: str, 
                    commit_message: str = "Upload Jojo LLM model") -> str:
        """Upload model to Hugging Face Hub"""
        
        try:
            self.logger.info(f"Uploading model from {local_dir} to {repo_name}")
            
            # Upload the entire folder
            url = upload_folder(
                folder_path=local_dir,
                repo_id=repo_name,
                repo_type="model",
                commit_message=commit_message
            )
            
            self.logger.info(f"Successfully uploaded model to: https://huggingface.co/{repo_name}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload model: {e}")
            raise e

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Upload Jojo LLM models to Hugging Face Hub")
    parser.add_argument("checkpoint", type=str, help="Path to Jojo checkpoint file")
    parser.add_argument("--repo-name", type=str, required=True, 
                       help="Name for the Hugging Face repository")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Name of the training dataset")
    parser.add_argument("--organization", type=str, default=None,
                       help="Hugging Face organization (optional)")
    parser.add_argument("--private", action="store_true",
                       help="Create private repository")
    parser.add_argument("--output-dir", type=str, default="./hf_model",
                       help="Local directory to save converted model")
    parser.add_argument("--tokenizer", type=str, choices=["gpt2", "extended"], 
                       default="gpt2", help="Tokenizer type to use")
    parser.add_argument("--dry-run", action="store_true",
                       help="Convert model but don't upload")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--commit-message", type=str, 
                       default="Upload Jojo LLM model",
                       help="Commit message for the upload")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    # Check dependencies
    if not HUGGINGFACE_AVAILABLE:
        logger.error("Hugging Face libraries not available. Install with:")
        logger.error("pip install transformers huggingface_hub")
        sys.exit(1)
    
    if not JOJO_MODULES_AVAILABLE:
        logger.error("Jojo modules not available. Make sure you're in the Jojo directory.")
        sys.exit(1)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    try:
        logger.info(f"{BOLD}{BLUE}╔══════════════════════════════════════════════════════════╗{ENDC}")
        logger.info(f"{BOLD}{BLUE}║             Jojo LLM → Hugging Face Uploader            ║{ENDC}")
        logger.info(f"{BOLD}{BLUE}╚══════════════════════════════════════════════════════════╝{ENDC}")
        
        # Initialize components
        converter = JojoToHuggingFaceConverter(logger)
        card_generator = ModelCardGenerator(logger)
        uploader = HuggingFaceUploader(logger)
        
        # Convert model
        logger.info(f"{CYAN}Step 1: Converting Jojo checkpoint to Hugging Face format...{ENDC}")
        model_info = converter.convert_model(args.checkpoint, args.output_dir)
        
        # Setup tokenizer
        logger.info(f"{CYAN}Step 2: Setting up tokenizer...{ENDC}")
        uploader.setup_tokenizer(args.output_dir, args.tokenizer)
        
        # Generate model card
        logger.info(f"{CYAN}Step 3: Generating model card...{ENDC}")
        model_card = card_generator.generate_model_card(
            model_info=model_info,
            model_name=args.repo_name,
            dataset_name=args.dataset,
            checkpoint_path=args.checkpoint,
            additional_info={"repo_name": args.repo_name}
        )
        
        # Save model card
        model_card_path = os.path.join(args.output_dir, "README.md")
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        logger.info(f"Saved model card to {model_card_path}")
        
        if args.dry_run:
            logger.info(f"{YELLOW}Dry run mode - model converted but not uploaded{ENDC}")
            logger.info(f"Converted model saved to: {args.output_dir}")
            logger.info(f"To upload manually, run:")
            logger.info(f"huggingface-cli upload {args.repo_name} {args.output_dir}")
        else:
            # Create repository and upload
            logger.info(f"{CYAN}Step 4: Creating Hugging Face repository...{ENDC}")
            full_repo_name = uploader.create_repository(
                args.repo_name, args.private, args.organization
            )
            
            logger.info(f"{CYAN}Step 5: Uploading model to Hugging Face Hub...{ENDC}")
            upload_url = uploader.upload_model(
                args.output_dir, full_repo_name, args.commit_message
            )
            
            logger.info(f"{BOLD}{GREEN}✅ Successfully uploaded model!{ENDC}")
            logger.info(f"{BOLD}Repository URL: {GREEN}https://huggingface.co/{full_repo_name}{ENDC}")
            
            # Cleanup option
            cleanup = input(f"\n{YELLOW}Delete local converted files? (y/N): {ENDC}")
            if cleanup.lower() == 'y':
                import shutil
                shutil.rmtree(args.output_dir)
                logger.info(f"Cleaned up local files: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info(f"\n{YELLOW}Upload cancelled by user{ENDC}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
