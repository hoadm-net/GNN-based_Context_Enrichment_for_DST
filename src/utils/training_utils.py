"""
Training Utilities for History-Aware GraphDST

Chứa các utilities cho training: checkpoint management, early stopping,
learning rate scheduling, và other training helpers
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
import numpy as np
from collections import defaultdict


class EarlyStopping:
    """
    Early stopping để prevent overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' cho loss, 'max' cho accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, metric_value: float) -> bool:
        """
        Check if we should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False
            
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            improved = metric_value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            
        self.early_stop = self.counter >= self.patience
        return self.early_stop


class CheckpointManager:
    """
    Manages model checkpoints: saving, loading, cleanup
    """
    
    def __init__(self, checkpoint_dir: str, keep_best_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # List of (metric_value, filepath) tuples
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       metric_value: float,
                       config: Dict[str, Any],
                       is_best: bool = False) -> str:
        """
        Save model checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            checkpoint_name = f"best_model_epoch_{epoch}_{timestamp}.pt"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
            
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            'config': config,
            'timestamp': timestamp
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update best checkpoints list
        if is_best:
            self.best_checkpoints.append((metric_value, checkpoint_path))
            self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by metric value
            
            # Remove old checkpoints if needed
            if len(self.best_checkpoints) > self.keep_best_n:
                _, old_checkpoint_path = self.best_checkpoints.pop(0)
                if os.path.exists(old_checkpoint_path):
                    os.remove(old_checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[-1][1]


class MetricsTracker:
    """
    Tracks training và validation metrics
    """
    
    def __init__(self):
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.epoch_times = []
        
    def update_train_metrics(self, metrics: Dict[str, float]):
        """Update training metrics cho current epoch"""
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
    
    def update_val_metrics(self, metrics: Dict[str, float]):
        """Update validation metrics cho current epoch"""  
        for key, value in metrics.items():
            self.val_metrics[key].append(value)
    
    def record_epoch_time(self, epoch_time: float):
        """Record time taken for epoch"""
        self.epoch_times.append(epoch_time)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics"""
        latest = {}
        
        if self.train_metrics:
            latest['train'] = {k: v[-1] for k, v in self.train_metrics.items()}
        if self.val_metrics:
            latest['val'] = {k: v[-1] for k, v in self.val_metrics.items()}
        if self.epoch_times:
            latest['epoch_time'] = self.epoch_times[-1]
            latest['avg_epoch_time'] = np.mean(self.epoch_times)
        
        return latest
    
    def save_to_file(self, filepath: str):
        """Save metrics history to file"""
        metrics_data = {
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'epoch_times': self.epoch_times
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)


class LearningRateScheduler:
    """
    Custom learning rate scheduler với warmup
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 warmup_steps: int, 
                 total_steps: int,
                 min_lr_ratio: float = 0.1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Decay phase: cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
            lr *= self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class GradientAccumulator:
    """
    Handles gradient accumulation cho large effective batch sizes
    """
    
    def __init__(self, accumulation_steps: int, max_grad_norm: float = 1.0):
        """
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        
    def backward_and_step(self, loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """
        Backward pass với gradient accumulation
        
        Returns:
            True if optimizer step was taken, False otherwise
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Check if we should step optimizer
        if self.current_step % self.accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            return True
        
        return False


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Returns:
        total_params, trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    
    # Simulate decreasing loss
    for i, loss in enumerate([1.0, 0.8, 0.75, 0.76, 0.77, 0.78]):
        should_stop = early_stopping(loss)
        print(f"Epoch {i+1}, Loss: {loss}, Should stop: {should_stop}")
        if should_stop:
            break
    
    print("Training utilities test completed!")