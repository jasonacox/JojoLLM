#!/usr/bin/python3
"""
Main Trainer class for Jojo LLM Training

This module provides the main training orchestration with improved
organization, efficiency, and monitoring capabilities.

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import os
import time
import logging
import datetime
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from config import Config, Constants
from utils import (
    ProgressTracker, MetricsTracker, GracefulShutdown, DeviceManager,
    CheckpointManager, PlotManager, format_time_delta, count_parameters
)
from data_loader import CachedJsonlDataset, EfficientDataLoader, create_dataloaders

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator"""
    
    def __init__(self, config: Config, model: nn.Module, tokenizer: Any):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize components
        self.metrics = MetricsTracker()
        self.shutdown_handler = GracefulShutdown()
        self.device = torch.device(config.system.device)
        
        # Training state
        self.epoch = 0
        self.batch_counter = 0
        self.best_val_loss = float('inf')
        self.worst_val_loss = float('-inf')
        self.best_train_loss = float('inf')
        self.worst_train_loss = float('-inf')
        
        # Initialize data loaders first
        self._setup_data_loaders()
        
        # Initialize optimizer and scheduler (needs data loaders)
        self._setup_optimizer_and_scheduler()
        
        # Initialize mixed precision training
        if 'cuda' in config.system.device:
            self.scaler = torch.amp.GradScaler('cuda', enabled=(config.system.dtype == 'float16'))
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=(config.system.dtype == 'float16'))
        
        # Autocast context
        device_type = 'cuda' if 'cuda' in config.system.device else 'cpu'
        dtype_map = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }
        self.autocast_ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=dtype_map[config.system.dtype]
        )
        
        logger.info("Trainer initialized successfully")
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """Initialize optimizer and learning rate scheduler"""
        # Configure optimizer
        self.optimizer = self.model.configure_optimizers(
            self.config.optimizer.weight_decay,
            self.config.optimizer.learning_rate,
            (self.config.optimizer.beta1, self.config.optimizer.beta2),
            'cuda' if 'cuda' in self.config.system.device else 'cpu'
        )
        
        # Setup learning rate scheduler
        if self.config.scheduler.decay_lr:
            # Use a simple cosine annealing scheduler
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            # Calculate total steps for cosine annealing
            total_steps = len(self.train_loader) * self.config.training.max_epochs
            
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.scheduler.min_lr
            )
        else:
            self.lr_scheduler = None
    
    def _setup_data_loaders(self) -> None:
        """Initialize data loaders"""
        # Construct file paths
        train_file = os.path.join(
            self.config.data.data_dir,
            f"{self.config.data.dataset_name}-train.jsonl"
        )
        val_file = os.path.join(
            self.config.data.data_dir,
            f"{self.config.data.dataset_name}-val.jsonl"
        )
        
        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            train_file, val_file, self.tokenizer,
            self.config.training.batch_size,
            self.config.model.block_size,
            self.config.data.cache_dir if self.config.data.cache_tokenized else None,
            self.device
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Create progress tracker
        progress_tracker = ProgressTracker(
            len(self.train_loader), self.epoch, self.config.training.max_epochs
        )
        
        # Training loop
        batch_idx = 0
        running_mfu = -1.0
        
        for batch_data in self.train_loader:
            if self.shutdown_handler.should_stop():
                logger.info("Graceful shutdown requested during training")
                break
            
            batch_idx += 1
            batch_start_time = time.time()
            
            # Get batch data
            X, Y = batch_data
            
            # Forward pass with gradient accumulation
            total_loss = 0.0
            
            # Gradient accumulation loop
            for micro_step in range(self.config.training.gradient_accumulation_steps):
                with self.autocast_ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.config.training.gradient_accumulation_steps
                    total_loss += loss.item()
                
                # Backward pass
                self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.optimizer.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.optimizer.grad_clip
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Calculate metrics
            batch_time = time.time() - batch_start_time
            samples_per_sec = self.config.training.batch_size / batch_time if batch_time > 0 else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update MFU (Model FLOPs Utilization)
            if self.batch_counter >= Constants.MFU_WARMUP_BATCHES:
                mfu = self.model.estimate_mfu(
                    self.config.training.batch_size * self.config.training.gradient_accumulation_steps,
                    batch_time
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            
            # Track metrics - Log training loss at every batch for plotting
            self.metrics.log_metric('train_loss_batch', total_loss, self.batch_counter)
            
            # Log additional metrics at specified intervals to avoid overwhelming logs
            if self.config.training.log_interval > 0 and self.batch_counter % self.config.training.log_interval == 0:
                self.metrics.log_metric('learning_rate', current_lr, self.batch_counter)
                self.metrics.log_metric('samples_per_sec', samples_per_sec, self.batch_counter)
                if running_mfu > 0:
                    self.metrics.log_metric('mfu', running_mfu, self.batch_counter)
                
                # Generate plot every log_interval batches
                #print(f"{Constants.CYAN}Generating plot at batch {self.batch_counter} (log_interval={self.config.training.log_interval}){Constants.ENDC}")
                self._generate_training_plot(f"Training Progress - Batch {self.batch_counter}")
            
            # Update running totals
            epoch_loss += total_loss
            num_batches += 1
            self.batch_counter += 1
            
            # Real-time progress display
            progress_line = progress_tracker.update(
                batch_idx, total_loss, current_lr, samples_per_sec, running_mfu
            )
            print(f"{progress_line}", flush=True)
            
            # Periodic evaluation during epoch
            if (self.config.training.eval_interval > 0 and 
                batch_idx % self.config.training.eval_interval == 0):
                
                print()  # New line for evaluation output
                eval_results = self.evaluate()
                
                # Log evaluation results
                print(f"\n{Constants.GREEN}Epoch {self.epoch+1} "
                      f"({batch_idx/len(self.train_loader)*100:.1f}%): "
                      f"Train Loss: {eval_results['train']:.4f}{Constants.ENDC}  "
                      f"{Constants.MAGENTA}Val Loss: {eval_results['val']:.4f}{Constants.ENDC}")
                
                # Track best and worst validation loss
                if eval_results['val'] < self.best_val_loss:
                    self.best_val_loss = eval_results['val']
                if eval_results['val'] > self.worst_val_loss:
                    self.worst_val_loss = eval_results['val']
                
                # Track best and worst training loss
                if eval_results['train'] < self.best_train_loss:
                    self.best_train_loss = eval_results['train']
                if eval_results['train'] > self.worst_train_loss:
                    self.worst_train_loss = eval_results['train']
                
                # Record evaluation metrics - Log at every evaluation for plotting
                self.metrics.log_metric('train_loss_eval', eval_results['train'], self.batch_counter)
                self.metrics.log_metric('val_loss_eval', eval_results['val'], self.batch_counter)
        
        # Epoch completion
        print()  # New line after progress bar
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Display epoch completion summary
        summary = progress_tracker.completion_summary(avg_epoch_loss, epoch_duration)
        print(summary)
        
        return {
            'avg_loss': avg_epoch_loss,
            'duration': epoch_duration,
            'num_batches': num_batches
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on train and validation sets"""
        self.model.eval()
        
        results = {}
        
        for split_name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            losses = []
            
            # Calculate appropriate number of evaluation batches
            eval_batches = min(
                self.config.training.eval_iters // self.config.training.batch_size,
                len(loader)
            )
            eval_batches = max(10, eval_batches)  # At least 10 batches
            
            for i, (X, Y) in enumerate(loader):
                if i >= eval_batches:
                    break
                
                with self.autocast_ctx:
                    logits, loss = self.model(X, Y)
                    losses.append(loss.item())
            
            results[split_name] = sum(losses) / len(losses) if losses else float('inf')
        
        self.model.train()
        return results
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> bool:
        """Save training checkpoint"""
        # Prepare checkpoint data
        checkpoint_data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'epoch': self.epoch,
            'batch_counter': self.batch_counter,
            'best_val_loss': self.best_val_loss,
            'worst_val_loss': self.worst_val_loss,
            'best_train_loss': self.best_train_loss,
            'worst_train_loss': self.worst_train_loss,
            'config': self.config.to_dict(),
            'metrics': dict(self.metrics.metrics),
            'metadata': CheckpointManager.create_checkpoint_metadata(self.config)
        }
        
        # Save checkpoint atomically
        success = CheckpointManager.save_checkpoint_atomic(checkpoint_data, filepath)
        
        if success:
            logger.info(f"Checkpoint saved: {filepath}")
            
            # Create loss curve plot
            self.plot_loss_curves(filepath)
            
            if is_best:
                # Also save as best model
                best_path = filepath.replace('.pt', '_best.pt')
                CheckpointManager.save_checkpoint_atomic(checkpoint_data, best_path)
                logger.info(f"Best model saved: {best_path}")
                # Create plot for best model too
                self.plot_loss_curves(best_path)
        else:
            logger.error(f"Failed to save checkpoint: {filepath}")
        
        return success
    
    def load_checkpoint(self, filepath: str, resume_training: bool = True) -> bool:
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model'])
            
            if resume_training:
                # Load training state
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler and checkpoint.get('lr_scheduler'):
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                self.epoch = checkpoint.get('epoch', 0)
                self.batch_counter = checkpoint.get('batch_counter', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                self.worst_val_loss = checkpoint.get('worst_val_loss', float('-inf'))
                self.best_train_loss = checkpoint.get('best_train_loss', float('inf'))
                self.worst_train_loss = checkpoint.get('worst_train_loss', float('-inf'))
                
                # Load metrics if available
                if 'metrics' in checkpoint:
                    for name, data in checkpoint['metrics'].items():
                        self.metrics.metrics[name] = data
            
            logger.info(f"Checkpoint loaded: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return False
    
    def train(self, checkpoint_path: Optional[str] = None, 
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        # Load checkpoint if resuming
        if resume_from:
            if not self.load_checkpoint(resume_from, resume_training=True):
                logger.error(f"Failed to load checkpoint for resuming: {resume_from}")
                return {'success': False, 'error': 'Failed to load checkpoint'}
        
        # Training setup
        start_time = time.time()
        checkpoint_path = checkpoint_path or f"models/{self.config.data.dataset_name}_epoch{self.config.training.max_epochs}.pt"
        
        logger.info(f"Starting training for {self.config.training.max_epochs} epochs")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Batches per epoch: {len(self.train_loader)}")
        
        try:
            # Training loop
            while self.epoch < self.config.training.max_epochs:
                if self.shutdown_handler.should_stop():
                    logger.info("Graceful shutdown requested")
                    break
                
                # Print epoch header
                self._print_epoch_header()
                
                # Train one epoch
                epoch_results = self.train_epoch()
                
                # End-of-epoch evaluation
                eval_results = self.evaluate()
                
                # Log results
                print(f"{Constants.GREEN}Train Loss: {eval_results['train']:.4f}{Constants.ENDC}  "
                      f"{Constants.MAGENTA}Val Loss: {eval_results['val']:.4f}{Constants.ENDC}")
                print(f"{Constants.YELLOW}Progress: {self.epoch+1}/{self.config.training.max_epochs} epochs "
                      f"({(self.epoch+1)/self.config.training.max_epochs*100:.1f}%){Constants.ENDC}\n")
                
                # Track metrics
                self.metrics.log_metric('train_loss_epoch', eval_results['train'], self.epoch)
                self.metrics.log_metric('val_loss_epoch', eval_results['val'], self.epoch)
                
                # Also log with standard names for plotting
                self.metrics.log_metric('train_loss', eval_results['train'], self.epoch)
                self.metrics.log_metric('val_loss', eval_results['val'], self.epoch)
                
                # Check if this is the best model and update all metrics
                is_best = eval_results['val'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = eval_results['val']
                if eval_results['val'] > self.worst_val_loss:
                    self.worst_val_loss = eval_results['val']
                
                # Update train loss tracking
                if eval_results['train'] < self.best_train_loss:
                    self.best_train_loss = eval_results['train']
                if eval_results['train'] > self.worst_train_loss:
                    self.worst_train_loss = eval_results['train']
                
                # Save checkpoint
                if self.config.training.save_checkpoints:
                    self.save_checkpoint(checkpoint_path, is_best=is_best)
                
                # Clear GPU cache
                if 'cuda' in self.config.system.device:
                    torch.cuda.empty_cache()
                
                # Move to next epoch
                self.epoch += 1
            
            # Training completion
            total_time = time.time() - start_time
            
            # Final evaluation
            final_eval = self.evaluate()
            
            # Update final metrics (in case this final evaluation is the best/worst)
            if final_eval['val'] < self.best_val_loss:
                self.best_val_loss = final_eval['val']
            if final_eval['val'] > self.worst_val_loss:
                self.worst_val_loss = final_eval['val']
            if final_eval['train'] < self.best_train_loss:
                self.best_train_loss = final_eval['train']
            if final_eval['train'] > self.worst_train_loss:
                self.worst_train_loss = final_eval['train']
            
            logger.info(f"Training completed in {format_time_delta(total_time)}")
            logger.info(f"Final train loss: {final_eval['train']:.4f}")
            logger.info(f"Final validation loss: {final_eval['val']:.4f}")
            logger.info(f"Best train loss: {self.best_train_loss:.4f}")
            logger.info(f"Worst train loss: {self.worst_train_loss:.4f}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Worst validation loss: {self.worst_val_loss:.4f}")
            
            return {
                'success': True,
                'final_train_loss': final_eval['train'],
                'final_val_loss': final_eval['val'],
                'best_train_loss': self.best_train_loss,
                'worst_train_loss': self.worst_train_loss,
                'best_val_loss': self.best_val_loss,
                'worst_val_loss': self.worst_val_loss,
                'total_time': total_time,
                'epochs_completed': self.epoch
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Save emergency checkpoint
            if self.config.training.save_checkpoints:
                emergency_path = checkpoint_path.replace('.pt', '_emergency.pt')
                self.save_checkpoint(emergency_path)
            
            return {
                'success': False,
                'error': str(e),
                'epochs_completed': self.epoch
            }
    
    def _print_epoch_header(self) -> None:
        """Print epoch header"""
        header_length = 42
        epoch_text = f"EPOCH {self.epoch+1} OF {self.config.training.max_epochs}"
        padding = " " * ((header_length - len(epoch_text)) // 2)
        right_padding = " " * (header_length - len(epoch_text) - len(padding))
        
        print(f"\n{Constants.BOLD}{Constants.BLUE}╔══════════════════════════════════════════╗{Constants.ENDC}")
        print(f"{Constants.BOLD}{Constants.BLUE}║{padding}{epoch_text}{right_padding}║{Constants.ENDC}")
        print(f"{Constants.BOLD}{Constants.BLUE}╚══════════════════════════════════════════╝{Constants.ENDC}\n")
    
    def plot_loss_curves(self, checkpoint_path: str) -> None:
        """Generate and save loss curve plots"""
        try:
            # Create plot filename
            plot_path = checkpoint_path.replace('.pt', '.png')
            
            # Generate plot title
            dataset_name = self.config.data.dataset_name
            title = f"Training Progress - {dataset_name} (Epoch {self.epoch+1})"
            
            # Generate the plot
            success = PlotManager.plot_training_curves(self.metrics, plot_path, title)
            
            if success:
                logger.info(f"Loss curve plot saved: {plot_path}")
            else:
                logger.warning("Could not generate loss curve plot")
                
        except Exception as e:
            logger.warning(f"Error generating loss curve plot: {e}")
    
    def _generate_training_plot(self, title: str) -> None:
        """Generate training plot during training (not just at checkpoints)"""
        try:
            # Create plot filename - use a consistent name that gets overwritten
            # Only save milestone plots every 100 batches to avoid too many files
            dataset_name = self.config.data.dataset_name
            
            plot_path = f"models/{dataset_name}.png"
            
            # Generate the plot
            from utils import PlotManager
            success = PlotManager.plot_training_curves(self.metrics, plot_path, title)
            
            if success:
                logger.info(f"Training plot saved: {plot_path}")
            else:
                logger.warning("Could not generate training plot")
                
        except Exception as e:
            logger.warning(f"Error generating training plot: {e}")
