"""
Pure Classification Delta Prediction Heads
All slots use classification approach (no span extraction)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json


class ClassificationDeltaPredictionHeads(nn.Module):
    """
    Pure classification prediction heads for all slots.
    Each slot has its own vocabulary-based classifier.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_slots: int,
                 slot_list: List[str],
                 slot_value_vocab: Dict[str, List[str]],
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_list = slot_list
        self.slot_value_vocab = slot_value_vocab
        
        # Shared heads for all slots
        self.slot_operation_head = nn.Linear(hidden_dim, num_slots * 4)  # KEEP/ADD/UPDATE/REMOVE
        self.value_existence_head = nn.Linear(hidden_dim, num_slots)     # Has value or not
        
        # Special value classifiers (all slots)
        self.none_classifier = nn.Linear(hidden_dim, num_slots)
        self.dontcare_classifier = nn.Linear(hidden_dim, num_slots)
        
        # Per-slot value classifiers
        self.value_classifiers = nn.ModuleDict()
        self.slot_vocab_sizes = {}
        
        for slot in slot_list:
            if slot in slot_value_vocab:
                vocab = slot_value_vocab[slot]
                if not vocab:
                    # Skip completely empty vocabularies; caller may populate later
                    continue
                # Filter special values
                real_values = [v for v in vocab if v.lower() not in ['none', 'dontcare', "don't care", '']]
                vocab_size = len(real_values)
                
                if vocab_size > 0:
                    self.slot_vocab_sizes[slot] = vocab_size
                    # Create classifier for this slot
                    slot_key = slot.replace('-', '_')
                    self.value_classifiers[slot_key] = nn.Linear(hidden_dim, vocab_size)
                else:
                    print(f"Warning: Slot {slot} has no valid values in vocabulary")
        
        # Create mappings
        self.slot2idx = {slot: idx for idx, slot in enumerate(slot_list)}
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ ClassificationDeltaPredictionHeads initialized:")
        print(f"   - Total slots: {num_slots}")
        print(f"   - Slots with classifiers: {len(self.value_classifiers)}")
        print(f"   - Vocab sizes: min={min(self.slot_vocab_sizes.values()) if self.slot_vocab_sizes else 0}, "
              f"max={max(self.slot_vocab_sizes.values()) if self.slot_vocab_sizes else 0}")
        
    def forward(self,
                pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pure classification
        
        Args:
            pooled_features: [batch, hidden_dim] - global representation
            
        Returns:
            Dictionary with all predictions
        """
        batch_size = pooled_features.size(0)
        
        # Apply dropout
        pooled_features = self.dropout(pooled_features)
        
        # === Shared Predictions (All Slots) ===
        
        # Slot operations: [batch, num_slots, 4]
        slot_ops_logits = self.slot_operation_head(pooled_features)
        slot_ops_logits = slot_ops_logits.view(batch_size, self.num_slots, 4)
        
        # Value existence: [batch, num_slots]
        value_existence_logits = self.value_existence_head(pooled_features)
        
        # Special values: [batch, num_slots]
        none_logits = self.none_classifier(pooled_features)
        dontcare_logits = self.dontcare_classifier(pooled_features)
        
        predictions = {
            'slot_operations': slot_ops_logits,
            'value_existence': value_existence_logits,
            'none': none_logits,
            'dontcare': dontcare_logits
        }
        
        # === Value Classification (All Slots) ===
        
        value_logits = {}
        for slot in self.slot_list:
            slot_key = slot.replace('-', '_')
            if slot_key in self.value_classifiers:
                logits = self.value_classifiers[slot_key](pooled_features)
                value_logits[slot] = logits  # [batch, vocab_size]
        
        predictions['values'] = value_logits
        
        return predictions
    
    def get_value_from_logits(self, 
                              slot: str, 
                              logits: torch.Tensor) -> str:
        """
        Convert classification logits to actual value
        
        Args:
            slot: Slot name
            logits: [vocab_size] logits
            
        Returns:
            Predicted value string
        """
        if slot not in self.slot_value_vocab:
            return "dontcare"
        
        vocab = self.slot_value_vocab[slot]
        real_values = [v for v in vocab if v.lower() not in ['none', 'dontcare', "don't care", '']]
        
        pred_idx = torch.argmax(logits).item()
        if pred_idx < len(real_values):
            return real_values[pred_idx]
        return "dontcare"


class ClassificationDeltaTargetComputer:
    """
    Compute delta targets for pure classification approach
    """
    
    def __init__(self, 
                 slot_list: List[str],
                 slot_value_vocab: Dict[str, List[str]]):
        self.slot_list = slot_list
        self.slot_value_vocab = slot_value_vocab
        self.slot2idx = {slot: idx for idx, slot in enumerate(slot_list)}
        self.num_slots = len(slot_list)
        
        # Build value to ID mappings
        self.value2id = {}
        for slot, vocab in slot_value_vocab.items():
            real_values = [v for v in vocab if v.lower() not in ['none', 'dontcare', "don't care", '']]
            self.value2id[slot] = {v.lower(): i for i, v in enumerate(real_values)}
        
        # Operation indices
        self.KEEP = 0
        self.ADD = 1
        self.UPDATE = 2
        self.REMOVE = 3
        
    def compute_delta_targets(self,
                             previous_belief_state: Dict[str, str],
                             current_belief_state: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Compute delta targets for classification approach
        
        Returns targets for:
        - Operations (all slots)
        - Value existence (all slots)
        - Value classification (per slot)
        - Special values (none/dontcare)
        """
        
        # Initialize labels
        operation_labels = torch.zeros(self.num_slots, dtype=torch.long)
        value_existence_labels = torch.zeros(self.num_slots)
        none_labels = torch.zeros(self.num_slots)
        dontcare_labels = torch.zeros(self.num_slots)
        
        # Value classification targets (sparse - only for changed/added slots)
        value_targets = {}  # slot -> target_id
        
        prev_set = set(previous_belief_state.keys()) if previous_belief_state else set()
        curr_set = set(current_belief_state.keys()) if current_belief_state else set()
        
        for slot_name in self.slot_list:
            slot_idx = self.slot2idx[slot_name]
            
            prev_has_slot = slot_name in prev_set
            curr_has_slot = slot_name in curr_set
            
            # Determine operation
            if not prev_has_slot and curr_has_slot:
                # ADD operation
                operation_labels[slot_idx] = self.ADD
                value_existence_labels[slot_idx] = 1.0
                curr_value = current_belief_state[slot_name].lower()
                
                # Check special values
                if curr_value == 'none':
                    none_labels[slot_idx] = 1.0
                elif curr_value in ["dontcare", "don't care"]:
                    dontcare_labels[slot_idx] = 1.0
                else:
                    # Regular value - get ID from vocabulary
                    if slot_name in self.value2id and curr_value in self.value2id[slot_name]:
                        value_targets[slot_name] = self.value2id[slot_name][curr_value]
                    
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
                    
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value in ["dontcare", "don't care"]:
                        dontcare_labels[slot_idx] = 1.0
                    else:
                        # Regular value
                        if slot_name in self.value2id and curr_value in self.value2id[slot_name]:
                            value_targets[slot_name] = self.value2id[slot_name][curr_value]
                else:
                    # KEEP operation
                    operation_labels[slot_idx] = self.KEEP
                    value_existence_labels[slot_idx] = 1.0
                    
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value in ["dontcare", "don't care"]:
                        dontcare_labels[slot_idx] = 1.0
            else:
                # No slot in both states
                operation_labels[slot_idx] = self.KEEP
                value_existence_labels[slot_idx] = 0.0
        
        return {
            'slot_operations': operation_labels,
            'value_existence': value_existence_labels,
            'none': none_labels,
            'dontcare': dontcare_labels,
            'value_targets': value_targets  # Dict[slot_name, target_id]
        }
