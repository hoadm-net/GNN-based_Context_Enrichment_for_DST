"""
Delta-aware Prediction Heads for DST

Implements prediction heads for delta belief state tracking:
- Slot operations: ADD/UPDATE/REMOVE/KEEP
- Value predictions: Categorical + span extraction
- Special tokens: None/DontCare handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

class DeltaPredictionHeads(nn.Module):
    """
    Delta-aware prediction heads for DST
    
    Predicts changes to belief state rather than absolute state
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_slots: int,
                 slot_list: List[str],
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_list = slot_list
        self.max_seq_len = max_seq_len
        
        # Slot operation prediction (KEEP=0, ADD=1, UPDATE=2, REMOVE=3)
        self.slot_operation_head = nn.Linear(hidden_dim, num_slots * 4)
        
        # Value existence prediction (has value or not) 
        self.value_existence_head = nn.Linear(hidden_dim, num_slots)
        
        # Span extraction heads for extractive values
        self.span_start_head = nn.Linear(hidden_dim, 1)
        self.span_end_head = nn.Linear(hidden_dim, 1) 
        
        # Special value classifiers
        self.none_classifier = nn.Linear(hidden_dim, num_slots)
        self.dontcare_classifier = nn.Linear(hidden_dim, num_slots)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                pooled_features: torch.Tensor,
                sequence_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for delta predictions
        
        Args:
            pooled_features: [batch, hidden_dim] - global representation
            sequence_features: [batch, seq_len, hidden_dim] - token representations
            attention_mask: [batch, seq_len] - attention mask for sequences
            
        Returns:
            Dictionary of predictions
        """
        batch_size = pooled_features.size(0)
        
        # Apply dropout
        pooled_features = self.dropout(pooled_features)
        
        # Slot operations: [batch, num_slots, 4] (KEEP/ADD/UPDATE/REMOVE)
        slot_ops_logits = self.slot_operation_head(pooled_features)
        slot_ops_logits = slot_ops_logits.view(batch_size, self.num_slots, 4)
        
        # Value existence: [batch, num_slots]
        value_existence_logits = self.value_existence_head(pooled_features)
        
        # Special value predictions: [batch, num_slots]
        none_logits = self.none_classifier(pooled_features)
        dontcare_logits = self.dontcare_classifier(pooled_features)
        
        return {
            'slot_operations': slot_ops_logits,
            'value_existence': value_existence_logits,
            'none': none_logits,
            'dontcare': dontcare_logits
        }
        
        # Span extraction (if sequence features provided)
        if sequence_features is not None:
            sequence_features = self.dropout(sequence_features)
            
            # Start and end logits: [batch, seq_len]
            start_logits = self.span_start_head(sequence_features).squeeze(-1)
            end_logits = self.span_end_head(sequence_features).squeeze(-1)
            
            # Apply attention mask (convert to boolean if needed)
            if attention_mask is not None:
                bool_mask = attention_mask.bool()
                start_logits = start_logits.masked_fill(~bool_mask, -float('inf'))
                end_logits = end_logits.masked_fill(~bool_mask, -float('inf'))
            
            predictions['span_start_logits'] = start_logits
            predictions['span_end_logits'] = end_logits
        
        return predictions


class DeltaTargetComputer:
    """
    Utility class to compute delta training targets from consecutive belief states
    """
    
    def __init__(self, slot_list: List[str]):
        self.slot_list = slot_list
        self.slot2idx = {slot: idx for idx, slot in enumerate(slot_list)}
        self.num_slots = len(slot_list)
        
        # Operation indices
        self.KEEP = 0
        self.ADD = 1  
        self.UPDATE = 2
        self.REMOVE = 3
        
    def compute_delta_targets(self, 
                            previous_belief_state: Dict[str, str],
                            current_belief_state: Dict[str, str],
                            slot_list: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute delta targets from consecutive belief states
        
        Args:
            previous_belief_state: Previous turn's belief state
            current_belief_state: Current turn's belief state
            slot_list: List of slot names (if None, use self.slot_list)
            
        Returns:
            Dictionary of delta targets
        """
        
        if slot_list is None:
            slot_list = self.slot_list
        
        # Initialize labels (batch size = 1 assumed) - use CPU by default, will be moved to GPU in trainer
        operation_labels = torch.zeros(len(slot_list), dtype=torch.long)
        value_existence_labels = torch.zeros(len(slot_list))
        none_labels = torch.zeros(len(slot_list))
        dontcare_labels = torch.zeros(len(slot_list))
        
        prev_set = set(previous_belief_state.keys()) if previous_belief_state else set()
        curr_set = set(current_belief_state.keys()) if current_belief_state else set()
        
        for slot_name in self.slot_list:
            slot_idx = self.slot2idx[slot_name]
            
            prev_has_slot = slot_name in prev_set
            curr_has_slot = slot_name in curr_set
            
            if not prev_has_slot and curr_has_slot:
                # ADD operation
                operation_labels[slot_idx] = self.ADD
                value_existence_labels[slot_idx] = 1.0
                
                # Check for special values
                curr_value = current_belief_state[slot_name].lower()
                if curr_value == 'none':
                    none_labels[slot_idx] = 1.0
                elif curr_value == "dontcare" or curr_value == "don't care":
                    dontcare_labels[slot_idx] = 1.0
                    
            elif prev_has_slot and not curr_has_slot:
                # REMOVE operation  
                operation_labels[slot_idx] = self.REMOVE
                value_existence_labels[slot_idx] = 0.0
                
            elif prev_has_slot and curr_has_slot:
                prev_value = previous_belief_state[slot_name].lower()
                curr_value = current_belief_state[slot_name].lower()
                
                if prev_value != curr_value:
                    # UPDATE operation
                    operation_labels[slot_idx] = self.UPDATE
                    value_existence_labels[slot_idx] = 1.0
                    
                    # Check for special values
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value == "dontcare" or curr_value == "don't care":
                        dontcare_labels[slot_idx] = 1.0
                else:
                    # KEEP operation (no change)
                    operation_labels[slot_idx] = self.KEEP
                    value_existence_labels[slot_idx] = 1.0
                    
                    # Maintain special value labels
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value == "dontcare" or curr_value == "don't care":
                        dontcare_labels[slot_idx] = 1.0
            else:
                # Both empty - KEEP with no value
                operation_labels[slot_idx] = self.KEEP
                value_existence_labels[slot_idx] = 0.0
                
        return {
            'slot_operations': operation_labels,
            'value_existence': value_existence_labels, 
            'none': none_labels,
            'dontcare': dontcare_labels
        }
    
    def delta_to_cumulative(self, 
                          prev_belief_state: Dict[str, str],
                          predicted_delta: Dict[str, torch.Tensor],
                          threshold: float = 0.5) -> Dict[str, str]:
        """
        Convert delta predictions to cumulative belief state
        
        Args:
            prev_belief_state: Previous belief state
            predicted_delta: Model's delta predictions
            threshold: Threshold for binary predictions
            
        Returns:
            Updated cumulative belief state
        """
        new_belief_state = prev_belief_state.copy() if prev_belief_state else {}
        
        # Get predictions
        slot_ops = predicted_delta['slot_operations']  # [num_slots, 4]
        value_existence = predicted_delta.get('value_existence', None)
        none_logits = predicted_delta.get('none_logits', None)
        dontcare_logits = predicted_delta.get('dontcare_logits', None)
        
        # Convert to probabilities
        slot_ops_probs = torch.softmax(slot_ops, dim=-1)
        operation_preds = torch.argmax(slot_ops_probs, dim=-1)  # [num_slots]
        
        if value_existence is not None:
            value_existence_probs = torch.sigmoid(value_existence)
            
        if none_logits is not None:
            none_probs = torch.sigmoid(none_logits)
            
        if dontcare_logits is not None:
            dontcare_probs = torch.sigmoid(dontcare_logits)
        
        for slot_idx, slot_name in enumerate(self.slot_list):
            predicted_op = operation_preds[slot_idx].item()
            
            if predicted_op == self.ADD or predicted_op == self.UPDATE:
                # Determine value
                value = "unknown"  # Default value
                
                # Check for special values
                if none_logits is not None and none_probs[slot_idx] > threshold:
                    value = "none"
                elif dontcare_logits is not None and dontcare_probs[slot_idx] > threshold:
                    value = "dontcare" 
                
                new_belief_state[slot_name] = value
                
            elif predicted_op == self.REMOVE:
                if slot_name in new_belief_state:
                    del new_belief_state[slot_name]
            
            # KEEP: no change needed
                
        return new_belief_state