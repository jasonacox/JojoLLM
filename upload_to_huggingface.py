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
            # Handle tie_word_embeddings for weight tying
            tie_word_embeddings=True,  # Jojo uses weight tying by default
        )
        
        self.logger.info(f"Created Hugging Face config: vocab_size={hf_config.vocab_size}, "
                        f"n_layer={hf_config.n_layer}, n_embd={hf_config.n_embd}, "
                        f"tie_word_embeddings={hf_config.tie_word_embeddings}")
        return hf_config
    
    def convert_state_dict(self, jojo_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert Jojo state dict to Hugging Face format"""
        
        self.logger.info("Converting Jojo state dict to Hugging Face format...")
        self.logger.debug(f"Jojo state dict keys: {list(jojo_state_dict.keys())}")
        
        hf_state_dict = {}
        unmapped_keys = set(jojo_state_dict.keys())
        
        # Helper function to clean key names (remove _orig_mod. prefix if present)
        def clean_key(key):
            if key.startswith('_orig_mod.'):
                return key[10:]  # Remove '_orig_mod.' prefix
            return key
        
        # Create clean mapping of Jojo keys
        clean_jojo_dict = {}
        for key, value in jojo_state_dict.items():
            clean_key_name = clean_key(key)
            clean_jojo_dict[clean_key_name] = value
        
        self.logger.debug(f"Clean Jojo state dict keys: {list(clean_jojo_dict.keys())}")
        
        # Direct mapping for exact matches (using clean keys)
        direct_mapping = {
            'transformer.wte.weight': 'transformer.wte.weight',
            'transformer.wpe.weight': 'transformer.wpe.weight',
            'transformer.ln_f.weight': 'transformer.ln_f.weight',
        }
        
        # Handle direct mappings
        converted_count = 0
        for hf_name, jojo_clean_name in direct_mapping.items():
            if jojo_clean_name in clean_jojo_dict:
                hf_state_dict[hf_name] = clean_jojo_dict[jojo_clean_name].clone()
                converted_count += 1
                self.logger.debug(f"Mapped {jojo_clean_name} -> {hf_name}")
                # Remove from unmapped keys
                for orig_key in jojo_state_dict.keys():
                    if clean_key(orig_key) == jojo_clean_name:
                        unmapped_keys.discard(orig_key)
        
        # Handle weight tying: lm_head shares weights with wte
        if 'transformer.wte.weight' in clean_jojo_dict:
            hf_state_dict['lm_head.weight'] = clean_jojo_dict['transformer.wte.weight'].clone()
            self.logger.debug("Applied weight tying: transformer.wte.weight -> lm_head.weight")
        elif 'lm_head.weight' in clean_jojo_dict:
            # Fallback if lm_head is separate (but Jojo uses weight tying)
            hf_state_dict['lm_head.weight'] = clean_jojo_dict['lm_head.weight'].clone()
            converted_count += 1
            self.logger.debug("Mapped lm_head.weight directly")
            for orig_key in jojo_state_dict.keys():
                if clean_key(orig_key) == 'lm_head.weight':
                    unmapped_keys.discard(orig_key)
        
        # Handle transformer blocks (with proper weight transposition)
        for orig_key, value in jojo_state_dict.items():
            clean_key_name = clean_key(orig_key)
            if clean_key_name.startswith('transformer.h.'):
                # Determine if this weight needs transposition
                if ('attn.c_attn.weight' in clean_key_name or 
                    'attn.c_proj.weight' in clean_key_name or
                    'mlp.c_fc.weight' in clean_key_name or 
                    'mlp.c_proj.weight' in clean_key_name):
                    # These linear layer weights need to be transposed
                    # Jojo: [out_features, in_features] -> HF: [in_features, out_features]
                    hf_state_dict[clean_key_name] = value.clone().transpose(0, 1)
                    self.logger.debug(f"Mapped and transposed: {clean_key_name} from {value.shape} to {value.transpose(0, 1).shape}")
                else:
                    # Layer norm weights don't need transposition
                    hf_state_dict[clean_key_name] = value.clone()
                    self.logger.debug(f"Mapped transformer block: {clean_key_name}")
                
                converted_count += 1
                unmapped_keys.discard(orig_key)
        
        self.logger.info(f"Converted {len(jojo_state_dict)} -> {converted_count} parameters")
        
        if unmapped_keys:
            self.logger.warning(f"Unmapped Jojo parameters: {unmapped_keys}")
        
        return hf_state_dict
        
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
            # First try strict loading
            missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
            
            if missing_keys:
                self.logger.warning(f"Missing keys in HF model: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in converted state dict: {unexpected_keys}")
            
            # Check if critical weights loaded correctly
            critical_weights = ['transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight']
            for weight_name in critical_weights:
                if weight_name in missing_keys:
                    self.logger.error(f"Critical weight missing: {weight_name}")
                else:
                    self.logger.debug(f"Successfully loaded: {weight_name}")
            
            self.logger.info("Successfully loaded converted weights into HF model")
            
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            raise e
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and config
        hf_model.save_pretrained(output_dir)
        self.logger.info(f"Saved HF model to {output_dir}")
        
        # Validate the converted model
        self._validate_converted_model(hf_model, hf_config, output_dir)
        
        return {
            'model_config': checkpoint_data['model_config'],
            'training_config': checkpoint_data['training_config'],
            'metadata': checkpoint_data['metadata'],
            'metrics': checkpoint_data['metrics'],
            'hf_config': hf_config
        }
    
    def _validate_converted_model(self, hf_model, hf_config, output_dir):
        """Validate that the converted model works correctly"""
        try:
            self.logger.info("Validating converted model...")
            
            # Test forward pass
            import torch
            test_input = torch.randint(0, hf_config.vocab_size, (1, 10))
            
            with torch.no_grad():
                outputs = hf_model(test_input)
                logits = outputs.logits
                
            self.logger.info(f"Model validation successful. Output shape: {logits.shape}")
            
            # Check for NaN or infinite values
            if torch.isnan(logits).any():
                self.logger.error("Model outputs contain NaN values!")
            elif torch.isinf(logits).any():
                self.logger.error("Model outputs contain infinite values!")
            else:
                self.logger.info("Model outputs are valid (no NaN/inf)")
                
            # Test that the model can generate reasonable probability distributions
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            self.logger.info(f"Top 5 token probabilities: {top_probs.tolist()}")
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            # Don't raise - validation failure shouldn't stop the conversion

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
        
        # Format vocabulary size with commas if it's a number
        vocab_size = model_config.get('vocab_size', 'N/A')
        vocab_size_str = f"{vocab_size:,}" if isinstance(vocab_size, (int, float)) else str(vocab_size)
        
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
- **Vocabulary Size**: {vocab_size_str} tokens
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
                repo_type="model",
                exist_ok=True  # Allow overwriting existing repos
            )
            self.logger.info(f"Created repository: {full_repo_name}")
            return full_repo_name
        except Exception as e:
            # Check for various "already exists" error messages
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in [
                "already exists", "already created", "conflict", 
                "you already created this model repo"
            ]):
                self.logger.info(f"Repository {full_repo_name} already exists - will overwrite")
                return full_repo_name
            else:
                self.logger.error(f"Failed to create repository: {e}")
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
                commit_message=commit_message,
                ignore_patterns=["*.git*", "*.DS_Store", "__pycache__/*"]
            )
            
            self.logger.info(f"Successfully uploaded model to: https://huggingface.co/{repo_name}")
            return url
            
        except Exception as e:
            error_msg = str(e)
            if "conflict" in error_msg.lower() or "409" in error_msg:
                self.logger.warning(f"Repository conflict detected. Attempting to overwrite...")
                try:
                    # Try uploading again with force
                    url = upload_folder(
                        folder_path=local_dir,
                        repo_id=repo_name,
                        repo_type="model",
                        commit_message=f"{commit_message} (overwrite)",
                        ignore_patterns=["*.git*", "*.DS_Store", "__pycache__/*"]
                    )
                    self.logger.info(f"Successfully overwrote model at: https://huggingface.co/{repo_name}")
                    return url
                except Exception as retry_error:
                    self.logger.error(f"Failed to overwrite model: {retry_error}")
                    raise retry_error
            else:
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
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing repository if it exists")
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
            if not args.overwrite:
                # Check if repo exists and warn user
                try:
                    from huggingface_hub import repo_exists
                    repo_name_to_check = f"{args.organization}/{args.repo_name}" if args.organization else args.repo_name
                    if repo_exists(repo_name_to_check, repo_type="model"):
                        logger.warning(f"Repository {repo_name_to_check} already exists!")
                        overwrite_choice = input(f"{YELLOW}Do you want to overwrite it? (y/N): {ENDC}")
                        if overwrite_choice.lower() != 'y':
                            logger.info("Upload cancelled by user")
                            sys.exit(0)
                except ImportError:
                    # If repo_exists is not available, proceed anyway
                    pass
            
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
