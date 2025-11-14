"""
Training script for Multi-Level GNN-based DST with Delta Prediction
Giữ nguyên architecture của UpdatedHistoryAwareGraphDST
Chỉ tạo training loop để handle batches
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np

from src.models.updated_history_aware_graphdst import UpdatedHistoryAwareGraphDST
from src.data.graph_dataloader import create_dataloaders


def compute_delta_loss(model_output,
                       model: UpdatedHistoryAwareGraphDST,
                       previous_belief_state: Dict[str, str],
                       current_belief_state: Dict[str, str]) -> torch.Tensor:
    """
    Compute loss cho delta prediction
    
    Args:
        model_output: DSTPrediction object từ model.forward()
        model: Model instance (for target computer)
        previous_belief_state: Previous belief state dict
        current_belief_state: Current belief state dict
    
    Returns:
        loss: Scalar tensor
    """
    # Extract predictions from DSTPrediction object
    delta_predictions = model_output.slot_predictions
    
    if not delta_predictions:
        device = next(model.parameters()).device
        return torch.tensor(0.0, requires_grad=True, device=device)
    
    slot_operations = delta_predictions.get('slot_operations')
    value_existence_logits = delta_predictions.get('value_existence')
    value_logits_dict = delta_predictions.get('values', {})
    none_logits = delta_predictions.get('none')
    dontcare_logits = delta_predictions.get('dontcare')
    
    if slot_operations is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, requires_grad=True, device=device)
    
    device = slot_operations.device
    targets = model.delta_target_computer.compute_delta_targets(
        previous_belief_state or {},
        current_belief_state or {}
    )
    
    operation_targets = targets['slot_operations'].to(device)
    value_existence_targets = targets['value_existence'].to(device)
    none_targets = targets['none'].to(device)
    dontcare_targets = targets['dontcare'].to(device)
    value_targets = targets.get('value_targets', {})
    
    # 1. Operation loss
    operation_loss = nn.CrossEntropyLoss()(
        slot_operations.squeeze(0),
        operation_targets
    )
    
    # 2. Value existence loss
    value_existence_loss = torch.tensor(0.0, device=device)
    if value_existence_logits is not None:
        value_existence_loss = nn.BCEWithLogitsLoss()(
            value_existence_logits.squeeze(0),
            value_existence_targets
        )
    
    # 3. Special value losses
    special_loss = torch.tensor(0.0, device=device)
    if none_logits is not None:
        special_loss = special_loss + nn.BCEWithLogitsLoss()(none_logits.squeeze(0), none_targets)
    if dontcare_logits is not None:
        special_loss = special_loss + nn.BCEWithLogitsLoss()(dontcare_logits.squeeze(0), dontcare_targets)
    
    # 4. Value classification loss
    value_loss = torch.tensor(0.0, device=device)
    num_value_preds = 0
    for slot_name, target_idx in value_targets.items():
        if slot_name in value_logits_dict:
            logits = value_logits_dict[slot_name]
            target = torch.tensor([target_idx], device=device, dtype=torch.long)
            value_loss = value_loss + nn.CrossEntropyLoss()(logits, target)
            num_value_preds += 1
    if num_value_preds > 0:
        value_loss = value_loss / num_value_preds
    
    total_loss = operation_loss + value_existence_loss + special_loss + value_loss
    return total_loss


def train_epoch(model: UpdatedHistoryAwareGraphDST,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        batch_loss = None  # Will accumulate tensor loss
        batch_size = batch['batch_size']
        num_valid = 0
        
        # Process each instance in batch (model handles single instance)
        for i in range(batch_size):
            utterance = batch['utterances'][i]
            dialogue_history = batch['dialogue_histories'][i]
            previous_belief = batch['previous_belief_states'][i]
            current_belief = batch['current_belief_states'][i]
            
            try:
                # Forward pass
                output = model(
                    utterance=utterance,
                    dialogue_history=dialogue_history,
                    previous_belief_state=previous_belief,
                    return_attention=False
                )
                
                # Compute loss
                loss = compute_delta_loss(
                    model_output=output,
                    model=model,
                    previous_belief_state=previous_belief,
                    current_belief_state=current_belief
                )
                
                # Accumulate loss
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss = batch_loss + loss
                num_valid += 1
                
            except Exception as e:
                print(f"\nError processing instance {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_valid > 0 and batch_loss is not None:
            batch_loss = batch_loss / num_valid
            
            # Backward pass
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model: UpdatedHistoryAwareGraphDST,
            eval_loader: DataLoader,
            device: torch.device) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # For Joint Goal Accuracy
    total_correct = 0
    total_instances = 0
    
    pbar = tqdm(eval_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            batch_size = batch['batch_size']
            
            for i in range(batch_size):
                utterance = batch['utterances'][i]
                dialogue_history = batch['dialogue_histories'][i]
                previous_belief = batch['previous_belief_states'][i]
                current_target = batch['current_belief_states'][i]
                
                try:
                    # Forward pass
                    output = model(
                        utterance=utterance,
                        dialogue_history=dialogue_history,
                        previous_belief_state=previous_belief,
                        return_attention=False
                    )
                    
                    # Compute loss
                    loss = compute_delta_loss(
                        model_output=output,
                        model=model,
                        previous_belief_state=previous_belief,
                        current_belief_state=current_target
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Check accuracy
                    predicted_belief = output.belief_state
                    
                    # Joint Goal Accuracy: all slots must match
                    if predicted_belief == current_target:
                        total_correct += 1
                    
                    total_instances += 1
                    
                except Exception as e:
                    print(f"\nError in evaluation: {e}")
                    continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    jga = total_correct / total_instances if total_instances > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'joint_goal_accuracy': jga,
        'num_instances': total_instances
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Level GNN-based DST")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs to train (fresh run)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--additional-epochs", type=int, default=None, help="Extra epochs to run after checkpoint epoch")
    parser.add_argument("--save-dir", type=str, default="checkpoints/graph_dst", help="Directory to store checkpoints")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    default_config = {
        'data_dir': 'data/processed_graph',
        'batch_size': 64,
        'learning_rate': 5e-5,
        'num_epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': args.save_dir,
        'fusion_type': 'multimodal',
        'use_temporal_reasoning': True
    }

    checkpoint = None
    start_epoch = 0
    best_jga = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_config = checkpoint.get('config', {})
        default_config.update(checkpoint_config)
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_jga = checkpoint.get('jga', 0.0)

    # Apply CLI overrides
    if args.learning_rate is not None:
        default_config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        default_config['batch_size'] = args.batch_size
    if args.device is not None:
        default_config['device'] = args.device
    if args.resume and args.additional_epochs is not None:
        default_config['num_epochs'] = start_epoch + args.additional_epochs

    config = default_config

    print("="*60)
    print("Multi-Level GNN-based DST Training")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Save dir: {config['save_dir']}")
    print("="*60)

    # Create dataloaders
    train_loader, dev_loader, test_loader, train_dataset = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )

    # Initialize model
    print("\nInitializing model...")
    model = UpdatedHistoryAwareGraphDST(
        hidden_dim=768,
        fusion_dim=768,
        num_domains=5,
        max_history_turns=20,
        num_gnn_layers=2,
        num_heads=8,
        dropout=0.1,
        fusion_type=config['fusion_type']
    )

    # Setup ontology
    print("Loading ontology...")
    model.setup_ontology(
        ontology_path=os.path.join(config['data_dir'], '../raw/ontology.json'),
        slot_meta_path=os.path.join(config['data_dir'], 'slot_meta.json'),
        slot_value_vocab=train_dataset.slot_value_vocab
    )

    model = model.to(config['device'])

    # Load checkpoint weights if needed
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from epoch {start_epoch}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    if checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Loaded optimizer state")

    # Ensure save dir exists
    os.makedirs(config['save_dir'], exist_ok=True)

    total_epochs = config['num_epochs']
    if total_epochs <= start_epoch:
        print(f"No epochs to run (start_epoch={start_epoch}, target={total_epochs}).")
        return

    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=config['device']
        )

        print(f"\nTrain Loss: {train_loss:.4f}")

        dev_metrics = evaluate(
            model=model,
            eval_loader=dev_loader,
            device=config['device']
        )

        print(f"Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"Dev JGA: {dev_metrics['joint_goal_accuracy']:.4f}")

        if dev_metrics['joint_goal_accuracy'] > best_jga:
            best_jga = dev_metrics['joint_goal_accuracy']
            checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'jga': best_jga,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (JGA: {best_jga:.4f})")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Dev JGA: {best_jga:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
