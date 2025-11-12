"""
GraphDST Training Script

This script handles training of the GraphDST model on the MultiWOZ 2.4 dataset
using the current data pipeline.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.graphdst import GraphDSTModel, GraphDSTConfig, create_graphdst_model
from utils.graphdst_loader import create_graphdst_dataloaders
from evaluation.evaluator import DSTEvaluator


def setup_logging(log_dir: str, model_name: str = "graphdst"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class GraphDSTTrainer:
    """Trainer for GraphDST model"""
    
    def __init__(self, model: GraphDSTModel, dataloaders: Dict[str, DataLoader],
                 config: Dict, logger: logging.Logger):
        """
        Initialize trainer
        
        Args:
            model: GraphDST model
            dataloaders: Dictionary of dataloaders
            config: Training configuration
            logger: Logger instance
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.logger = logger
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        total_steps = len(dataloaders['train']) * config.get('num_epochs', 10)
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup evaluator
        slot_meta = self._load_slot_meta(config['slot_meta_path'])
        self.evaluator = DSTEvaluator(slot_meta)
        
        # Training state
        self.best_jga = 0.0
        self.current_epoch = 0
        
        # Loss weights
        self.loss_weights = config.get('loss_weights', {
            'domain': 1.0,
            'slot': 1.0,
            'span_start': 0.5,
            'span_end': 0.5
        })
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {warmup_steps}")
    
    def _load_slot_meta(self, slot_meta_path: str) -> List[str]:
        """Load slot metadata"""
        with open(slot_meta_path, 'r') as f:
            slot_data = json.load(f)
            return slot_data['slot_meta']
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'domain': 0.0, 'slot': 0.0, 'span_start': 0.0, 'span_end': 0.0}
        num_batches = len(self.dataloaders['train'])
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch['labels'].items()}
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Compute loss
            losses = self.model.compute_loss(predictions, labels)
            
            # Apply loss weights
            weighted_loss = 0.0
            for loss_name, loss_value in losses.items():
                if loss_name != 'total' and loss_name in self.loss_weights:
                    weighted_loss += loss_value * self.loss_weights[loss_name]
            
            if weighted_loss == 0.0:
                weighted_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name] += loss_value.item()
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{weighted_loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'domain': f"{losses.get('domain', torch.tensor(0.0)).item():.3f}",
                'slot': f"{losses.get('slot', torch.tensor(0.0)).item():.3f}"
            })
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        return epoch_losses
    
    def evaluate(self, split: str = 'dev') -> Dict[str, float]:
        """Evaluate model on specified split"""
        if split not in self.dataloaders:
            self.logger.warning(f"Split '{split}' not available for evaluation")
            return {}
        
        self.model.eval()
        all_predictions = []
        all_ground_truth = []
        
        self.logger.info(f"Evaluating on {split} set...")
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc=f"Evaluating {split}"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask)
                
                # Convert predictions to evaluation format
                batch_predictions = self._predictions_to_eval_format(predictions, batch)
                batch_ground_truth = self._labels_to_eval_format(batch)
                
                all_predictions.extend(batch_predictions)
                all_ground_truth.extend(batch_ground_truth)
        
        # Compute metrics
        if all_predictions and all_ground_truth:
            metrics = self.evaluator.evaluate_predictions(all_predictions, all_ground_truth)
            return metrics
        else:
            self.logger.warning("No predictions generated for evaluation")
            return {}
    
    def _predictions_to_eval_format(self, predictions: Dict[str, torch.Tensor], 
                                   batch: Dict) -> List[Dict]:
        """Convert model predictions to evaluation format"""
        batch_size = predictions['domains'].size(0)
        batch_predictions = []
        
        for i in range(batch_size):
            pred_dict = {
                'dialogue_id': batch['dialogue_ids'][i],
                'turn_id': batch['turn_ids'][i],
                'belief_state': []
            }
            
            # Process slot predictions
            for slot_name in self.model.slot_names:
                slot_key = slot_name.replace('-', '_').replace(' ', '_')
                
                if slot_name in predictions['slot_activations']:
                    slot_logits = predictions['slot_activations'][slot_name][i]  # (2,)
                    slot_prob = torch.softmax(slot_logits, dim=0)[1].item()  # Probability of active
                    
                    # Threshold for slot activation
                    if slot_prob > 0.5:
                        # For now, use a dummy value (in real implementation, 
                        # would extract from span predictions or use categorical heads)
                        pred_dict['belief_state'].append([slot_name, "dummy_value"])
            
            batch_predictions.append(pred_dict)
        
        return batch_predictions
    
    def _labels_to_eval_format(self, batch: Dict) -> List[Dict]:
        """Convert labels to evaluation format"""
        batch_size = batch['input_ids'].size(0)
        batch_ground_truth = []
        
        for i in range(batch_size):
            gt_dict = {
                'dialogue_id': batch['dialogue_ids'][i],
                'turn_id': batch['turn_ids'][i],
                'belief_state': []
            }
            
            # Extract active slots from labels
            for slot_name in self.model.slot_names:
                label_key = f"{slot_name}_active"
                if label_key in batch['labels']:
                    is_active = batch['labels'][label_key][i].item()
                    if is_active == 1:
                        gt_dict['belief_state'].append([slot_name, "dummy_value"])
            
            batch_ground_truth.append(gt_dict)
        
        return batch_ground_truth
    
    def save_checkpoint(self, save_dir: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_jga': self.best_jga,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if improved
        current_jga = metrics.get('joint_goal_accuracy', 0.0)
        if current_jga > self.best_jga:
            self.best_jga = current_jga
            best_path = os.path.join(save_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best JGA: {current_jga:.4f} - saved to {best_path}")
    
    def train(self, num_epochs: int, save_dir: str, eval_steps: int = 1):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_losses = self.train_epoch()
            
            # Log training metrics
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Losses:")
            for loss_name, loss_value in train_losses.items():
                self.logger.info(f"  {loss_name}: {loss_value:.4f}")
            
            # Evaluate periodically
            if (epoch + 1) % eval_steps == 0:
                eval_metrics = self.evaluate('dev')
                
                if eval_metrics:
                    self.logger.info(f"Epoch {epoch + 1} - Evaluation Metrics:")
                    for metric_name, metric_value in eval_metrics.items():
                        self.logger.info(f"  {metric_name}: {metric_value:.4f}")
                    
                    # Save checkpoint
                    self.save_checkpoint(save_dir, eval_metrics)
                else:
                    self.logger.warning("No evaluation metrics computed")
        
        # Final evaluation on test set
        if 'test' in self.dataloaders:
            self.logger.info("Final evaluation on test set...")
            test_metrics = self.evaluate('test')
            
            if test_metrics:
                self.logger.info("Final Test Results:")
                for metric_name, metric_value in test_metrics.items():
                    self.logger.info(f"  {metric_name}: {metric_value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train GraphDST model")
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--slot_meta_path', type=str, default='data/processed/slot_meta.json',
                       help='Path to slot metadata file')
    parser.add_argument('--output_dir', type=str, default='results/graphdst',
                       help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--eval_steps', type=int, default=1,
                       help='Evaluate every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 70)
    logger.info("GraphDST Model Training")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    
    # Create model
    logger.info("Creating GraphDST model...")
    config = GraphDSTConfig(
        hidden_dim=768,
        num_gnn_layers=3,
        num_attention_heads=8,
        dropout=0.1,
        learning_rate=args.learning_rate
    )
    
    model = create_graphdst_model(args.slot_meta_path, config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_graphdst_dataloaders(
        data_dir=args.data_dir,
        slot_meta_path=args.slot_meta_path,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Training configuration
    training_config = {
        'device': device,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'num_epochs': args.num_epochs,
        'slot_meta_path': args.slot_meta_path,
        'loss_weights': {
            'domain': 1.0,
            'slot': 2.0,  # Emphasize slot prediction
            'span_start': 0.5,
            'span_end': 0.5
        }
    }
    
    # Create trainer
    trainer = GraphDSTTrainer(
        model=model,
        dataloaders=dataloaders,
        config=training_config,
        logger=logger
    )
    
    # Start training
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
        eval_steps=args.eval_steps
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()