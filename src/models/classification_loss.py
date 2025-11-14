"""
Multi-Task Loss for Pure Classification DST
Combines operation prediction, value classification, and special value detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ClassificationMultiTaskLoss(nn.Module):
    """
    Multi-task loss for pure classification approach
    
    Components:
    1. Operation Loss: KEEP/ADD/UPDATE/REMOVE (weighted CE)
    2. Value Classification Loss: Per-slot value prediction (CE)
    3. Special Value Loss: None/DontCare detection (BCE)
    4. Value Existence Loss: Whether slot has value (BCE)
    """
    
    def __init__(self,
                 num_slots: int,
                 # Loss weights
                 alpha_operation: float = 1.0,
                 alpha_value: float = 2.0,
                 alpha_special: float = 0.5,
                 alpha_existence: float = 0.3,
                 # Operation class weights (handle imbalance)
                 operation_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.num_slots = num_slots
        
        # Loss component weights
        self.alpha_operation = alpha_operation
        self.alpha_value = alpha_value
        self.alpha_special = alpha_special
        self.alpha_existence = alpha_existence
        
        # Handle operation class imbalance
        if operation_weights is None:
            # Default weights: [KEEP, ADD, UPDATE, REMOVE]
            # KEEP is common (~70%), others are rare, so weight them more
            operation_weights = torch.tensor([0.5, 2.0, 2.0, 3.0])
        self.register_buffer('operation_weights', operation_weights)
        
        print(f"✅ ClassificationMultiTaskLoss initialized:")
        print(f"   - Operation weight: {alpha_operation}")
        print(f"   - Value weight: {alpha_value}")
        print(f"   - Special value weight: {alpha_special}")
        print(f"   - Existence weight: {alpha_existence}")
        print(f"   - Operation class weights: {operation_weights.tolist()}")
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions containing:
                - slot_operations: [batch, num_slots, 4]
                - values: Dict[slot_name, [batch, vocab_size]]
                - none: [batch, num_slots]
                - dontcare: [batch, num_slots]
                - value_existence: [batch, num_slots]
            
            targets: Ground truth containing:
                - slot_operations: [batch, num_slots]
                - value_targets: Dict[slot_name, [batch]] (sparse)
                - none: [batch, num_slots]
                - dontcare: [batch, num_slots]
                - value_existence: [batch, num_slots]
        
        Returns:
            Dictionary with total loss and component losses
        """
        
        batch_size = predictions['slot_operations'].size(0)
        device = predictions['slot_operations'].device
        
        # ===== 1. Operation Loss (Weighted CE for imbalance) =====
        
        # Reshape: [batch * num_slots, 4] and [batch * num_slots]
        operation_logits = predictions['slot_operations'].view(-1, 4)  # [B*S, 4]
        operation_targets = targets['slot_operations'].view(-1).long()  # [B*S]
        
        operation_loss = F.cross_entropy(
            operation_logits,
            operation_targets,
            weight=self.operation_weights.to(device),
            reduction='mean'
        )
        
        # ===== 2. Value Classification Loss (Per-slot CE) =====
        
        value_loss = torch.tensor(0.0, device=device)
        value_count = 0
        
        if 'value_targets' in targets and targets['value_targets']:
            pred_values = predictions['values']
            target_values = targets['value_targets']
            
            # Only compute loss for slots that have targets (ADD/UPDATE operations)
            for slot_name, target_ids in target_values.items():
                if slot_name in pred_values:
                    logits = pred_values[slot_name]  # [batch, vocab_size]
                    targets_slot = target_ids.long()  # [batch]
                    
                    # Filter out -1 (no target for this instance)
                    valid_mask = targets_slot >= 0
                    if valid_mask.any():
                        value_loss += F.cross_entropy(
                            logits[valid_mask],
                            targets_slot[valid_mask],
                            reduction='mean'
                        )
                        value_count += 1
            
            # Average over all slots with targets
            if value_count > 0:
                value_loss = value_loss / value_count
        
        # ===== 3. Special Value Loss (BCE) =====
        
        none_loss = F.binary_cross_entropy_with_logits(
            predictions['none'],
            targets['none'].float(),
            reduction='mean'
        )
        
        dontcare_loss = F.binary_cross_entropy_with_logits(
            predictions['dontcare'],
            targets['dontcare'].float(),
            reduction='mean'
        )
        
        special_loss = (none_loss + dontcare_loss) / 2.0
        
        # ===== 4. Value Existence Loss (BCE) =====
        
        existence_loss = F.binary_cross_entropy_with_logits(
            predictions['value_existence'],
            targets['value_existence'].float(),
            reduction='mean'
        )
        
        # ===== Total Loss =====
        
        total_loss = (
            self.alpha_operation * operation_loss +
            self.alpha_value * value_loss +
            self.alpha_special * special_loss +
            self.alpha_existence * existence_loss
        )
        
        # Return all components for monitoring
        return {
            'total': total_loss,
            'operation': operation_loss.item(),
            'value': value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0,
            'special': special_loss.item(),
            'existence': existence_loss.item(),
            'none': none_loss.item(),
            'dontcare': dontcare_loss.item()
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss component weights"""
        return {
            'operation': self.alpha_operation,
            'value': self.alpha_value,
            'special': self.alpha_special,
            'existence': self.alpha_existence
        }
    
    def set_loss_weights(self, **kwargs):
        """Dynamically adjust loss weights"""
        if 'operation' in kwargs:
            self.alpha_operation = kwargs['operation']
        if 'value' in kwargs:
            self.alpha_value = kwargs['value']
        if 'special' in kwargs:
            self.alpha_special = kwargs['special']
        if 'existence' in kwargs:
            self.alpha_existence = kwargs['existence']
        
        print(f"Updated loss weights: operation={self.alpha_operation}, "
              f"value={self.alpha_value}, special={self.alpha_special}, "
              f"existence={self.alpha_existence}")
