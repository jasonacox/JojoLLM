#!/usr/bin/python3
"""
Utility classes and functions for Jojo LLM Training

This module provides utility classes for progress tracking, metrics collection,
tensor management, and other common training utilities.

Author: Jason A. Cox
2025 July 4
https://github.com/jasonacox/jojo
"""

import time
import datetime
import logging
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any
import torch
from config import Constants


class ProgressTracker:
    """Real-time progress tracking for training"""
    
    def __init__(self, total_batches: int, epoch: int, max_epochs: int):
        self.total_batches = total_batches
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.start_time = time.time()
        
    def update(self, batch_idx: int, loss: float, lr: float, 
               samples_per_sec: float, mfu: Optional[float] = None) -> str:
        """Update progress and return formatted progress string"""
        
        # Calculate progress percentage
        progress = batch_idx / self.total_batches * 100
        bar_length = Constants.PROGRESS_BAR_LENGTH
        bar_filled = int(bar_length * batch_idx / self.total_batches)
        progress_bar = '█' * bar_filled + '░' * (bar_length - bar_filled)
        
        # Calculate ETA
        if batch_idx > 0:
            time_per_batch = (time.time() - self.start_time) / batch_idx
            eta_seconds = time_per_batch * (self.total_batches - batch_idx)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "N/A"
        
        # Build progress line
        progress_line = (
            f"{Constants.YELLOW}[{progress_bar}]{Constants.ENDC} "
            f"{Constants.BOLD}Epoch {self.epoch+1}/{self.max_epochs}{Constants.ENDC} | "
            f"{Constants.CYAN}Batch {batch_idx}/{self.total_batches}{Constants.ENDC} | "
            f"{Constants.MAGENTA}{progress:.1f}%{Constants.ENDC} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"{Constants.GREEN}ETA: {eta_str}{Constants.ENDC} | "
            f"Samples/s: {samples_per_sec:.1f}"
        )
        
        if mfu is not None and mfu > 0:
            progress_line += f" | MFU: {mfu*100:.1f}%"
            
        return progress_line
    
    def completion_summary(self, avg_loss: float, duration: float) -> str:
        """Generate epoch completion summary"""
        progress_bar = '█' * Constants.PROGRESS_BAR_LENGTH
        duration_str = str(datetime.timedelta(seconds=int(duration)))
        
        summary = (
            f"{Constants.YELLOW}[{progress_bar}]{Constants.ENDC} "
            f"{Constants.BOLD}Epoch {self.epoch+1}/{self.max_epochs}{Constants.ENDC} | "
            f"{Constants.CYAN}Batch {self.total_batches}/{self.total_batches}{Constants.ENDC} | "
            f"{Constants.MAGENTA}100.0%{Constants.ENDC} | "
            f"Loss: {avg_loss:.4f} | "
            f"{Constants.GREEN}Complete!{Constants.ENDC}\n\n"
            f"{Constants.BOLD}{Constants.GREEN}==== Epoch {self.epoch+1}/{self.max_epochs} Complete ===={Constants.ENDC}\n"
            f"{Constants.CYAN}Duration: {duration_str}{Constants.ENDC}\n"
            f"{Constants.CYAN}Average Loss: {avg_loss:.4f}{Constants.ENDC}"
        )
        
        return summary


class MetricsTracker:
    """Track and manage training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_counters = defaultdict(int)
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value"""
        if step is None:
            step = self.step_counters[name]
            self.step_counters[name] += 1
        
        self.metrics[name].append((step, value))
        
    def get_metric_history(self, name: str) -> List[Tuple[int, float]]:
        """Get full history of a metric"""
        return self.metrics[name]
    
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get the latest value of a metric"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1][1]
        return None
    
    def get_best_metric(self, name: str, minimize: bool = True) -> Optional[float]:
        """Get the best value of a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = [value for _, value in self.metrics[name]]
        return min(values) if minimize else max(values)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
            
        values = [value for _, value in self.metrics[name]]
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'count': len(values),
            'latest': values[-1]
        }
    
    def get_steps_and_values(self, name: str) -> Tuple[List[int], List[float]]:
        """Get steps and values for plotting"""
        if name not in self.metrics:
            return [], []
            
        data = self.metrics[name]
        steps = [step for step, _ in data]
        values = [value for _, value in data]
        return steps, values


class TensorBuffer:
    """Reusable tensor buffers to reduce memory allocation"""
    
    def __init__(self, batch_size: int, block_size: int, device: torch.device):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        # Pre-allocate buffers
        self.x_buffer = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        self.y_buffer = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
    
    def get_buffers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get clean tensor buffers"""
        # Zero out the buffers (faster than creating new tensors)
        self.x_buffer.zero_()
        self.y_buffer.zero_()
        return self.x_buffer, self.y_buffer
    
    def resize_if_needed(self, new_batch_size: int, new_block_size: int) -> None:
        """Resize buffers if needed"""
        if new_batch_size != self.batch_size or new_block_size != self.block_size:
            self.batch_size = new_batch_size
            self.block_size = new_block_size
            self.x_buffer = torch.zeros((new_batch_size, new_block_size), dtype=torch.long, device=self.device)
            self.y_buffer = torch.zeros((new_batch_size, new_block_size), dtype=torch.long, device=self.device)


class GracefulShutdown:
    """Handle graceful shutdown on interruption"""
    
    def __init__(self):
        self.shutdown_requested = False
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown_requested = True
        print(f"\n{Constants.YELLOW}Graceful shutdown requested... Saving checkpoint...{Constants.ENDC}")
    
    def should_stop(self) -> bool:
        """Check if shutdown was requested"""
        return self.shutdown_requested


class DeviceManager:
    """Manage device selection and memory optimization"""
    
    @staticmethod
    def select_best_device() -> str:
        """Automatically select the best available device"""
        if not torch.cuda.is_available():
            return 'cpu'
        
        # Find device with most free memory
        device_free_memory = []
        max_free_memory = 0
        best_device = 0
        
        print("CUDA devices available:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # Free memory in GB
            mem_total = props.total_memory / (1024 ** 3)  # Total memory in GB
            device_free_memory.append(mem_free)
            
            # Track device with most free memory
            if mem_free > max_free_memory:
                max_free_memory = mem_free
                best_device = i
                
            print(f"  [{i}] {props.name} - Free: {mem_free:.2f} GB / Total: {mem_total:.2f} GB")
        
        return f'cuda:{best_device}'
    
    @staticmethod
    def optimize_memory(device: str) -> None:
        """Optimize memory usage for the given device"""
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)


class CheckpointManager:
    """Manage checkpoint saving and loading with metadata"""
    
    @staticmethod
    def create_checkpoint_metadata(config: Any, git_commit: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive checkpoint metadata"""
        from config import Constants
        return {
            'trainer_version': Constants.VERSION,
            'trainer_version_name': Constants.VERSION_NAME,
            'pytorch_version': torch.__version__,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
            'git_commit': git_commit or CheckpointManager._get_git_commit(),
        }
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def save_checkpoint_atomic(checkpoint_data: Dict[str, Any], filepath: str) -> bool:
        """Save checkpoint atomically to prevent corruption"""
        import os
        
        temp_filepath = filepath + Constants.CHECKPOINT_TEMP_SUFFIX
        
        try:
            # Save to temporary file first
            torch.save(checkpoint_data, temp_filepath)
            
            # Atomic rename (on Unix systems)
            if os.name == 'posix':
                os.replace(temp_filepath, filepath)
            else:
                # On Windows, remove destination first (not atomic)
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            except:
                pass
            return False


class Logger:
    """Enhanced logging utility"""
    
    @staticmethod
    def setup_logging(debug: bool = False, log_file: str = Constants.LOG_FILE) -> None:
        """Set up logging configuration"""
        log_level = logging.DEBUG if debug else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            f'{Constants.BLUE}%(asctime)s{Constants.ENDC} '
            f'{Constants.GREEN}%(levelname)s:{Constants.ENDC} %(message)s'
        )
        
        # Set up root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log_file is provided)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            # Use simpler format for file logging (no colors)
            file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)


def format_time_delta(seconds: float) -> str:
    """Format time delta in human-readable format"""
    return str(datetime.timedelta(seconds=int(seconds)))


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 ** 2)
