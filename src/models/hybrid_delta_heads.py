"""
Hybrid Delta Prediction Heads
Combines classification (for slots with few values) and span extraction (for slots with many values)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json


class HybridDeltaPredictionHeads(nn.Module):
    """
    Hybrid prediction heads using:
    - Classification for slots with ≤10 unique values (15 slots)
    - Span extraction for slots with >10 unique values (16 slots)
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_slots: int,
                 slot_list: List[str],
                 slot_value_vocab: Dict[str, List[str]],
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_list = slot_list
        self.max_seq_len = max_seq_len
        
        # Load slot categorization
        self.classification_slots = [
            'hotel-internet', 'hospital-department', 'hotel-parking', 
            'hotel-type', 'restaurant-pricerange', 'hotel-pricerange',
            'hotel-area', 'attraction-area', 'restaurant-area',
            'hotel-book people', 'train-day', 'hotel-stars',
            'restaurant-book people', 'hotel-book stay', 'restaurant-book day'
        ]
        
        self.span_extraction_slots = [
            'train-book people', 'hotel-book day', 'train-destination',
            'train-departure', 'attraction-type', 'restaurant-book time',
            'hotel-name', 'taxi-arriveby', 'taxi-leaveat',
            'restaurant-food', 'train-arriveby', 'attraction-name',
            'train-leaveat', 'restaurant-name', 'taxi-departure',
            'taxi-destination'
        ]
        
        # Shared heads for all slots
        self.slot_operation_head = nn.Linear(hidden_dim, num_slots * 4)
        self.value_existence_head = nn.Linear(hidden_dim, num_slots)
        
        # Special value classifiers (all slots)
        self.none_classifier = nn.Linear(hidden_dim, num_slots)
        self.dontcare_classifier = nn.Linear(hidden_dim, num_slots)
        
        # Classification heads (for slots with few values)
        self.classification_heads = nn.ModuleDict()
        self.slot_vocab_sizes = {}
        
        for slot in self.classification_slots:
            if slot in slot_value_vocab:
                vocab = slot_value_vocab[slot]
                # Filter out [NONE] and special tokens
                real_values = [v for v in vocab if v not in ['[NONE]', 'none', '']]
                vocab_size = len(real_values)
                self.slot_vocab_sizes[slot] = vocab_size
                # Create classifier for this slot
                self.classification_heads[slot.replace('-', '_')] = nn.Linear(hidden_dim, vocab_size)
        
        # Span extraction heads (for slots with many values)
        self.span_start_head = nn.Linear(hidden_dim, 1)
        self.span_end_head = nn.Linear(hidden_dim, 1)
        
        # Which slot does the span belong to? (16-way classification)
        self.span_slot_head = nn.Linear(hidden_dim, len(self.span_extraction_slots))
        
        # Create mapping
        self.slot2idx = {slot: idx for idx, slot in enumerate(slot_list)}
        self.span_slot2idx = {slot: idx for idx, slot in enumerate(self.span_extraction_slots)}
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                pooled_features: torch.Tensor,
                sequence_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hybrid prediction
        
        Args:
            pooled_features: [batch, hidden_dim] - global representation
            sequence_features: [batch, seq_len, hidden_dim] - token representations
            attention_mask: [batch, seq_len] - attention mask
            
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
        
        # === Classification Branch (15 slots) ===
        
        classification_logits = {}
        for slot in self.classification_slots:
            slot_key = slot.replace('-', '_')
            if slot_key in self.classification_heads:
                logits = self.classification_heads[slot_key](pooled_features)
                classification_logits[slot] = logits  # [batch, vocab_size]
        
        predictions['classification'] = classification_logits
        
        # === Span Extraction Branch (16 slots) ===
        
        if sequence_features is not None:
            sequence_features = self.dropout(sequence_features)
            
            # Start and end logits: [batch, seq_len]
            start_logits = self.span_start_head(sequence_features).squeeze(-1)
            end_logits = self.span_end_head(sequence_features).squeeze(-1)
            
            # Apply attention mask
            if attention_mask is not None:
                bool_mask = attention_mask.bool()
                start_logits = start_logits.masked_fill(~bool_mask, -float('inf'))
                end_logits = end_logits.masked_fill(~bool_mask, -float('inf'))
            
            # Which slot does the span belong to?
            span_slot_logits = self.span_slot_head(pooled_features)  # [batch, 16]
            
            predictions['span_start'] = start_logits
            predictions['span_end'] = end_logits
            predictions['span_slot'] = span_slot_logits
        
        return predictions
    
    def get_value_from_classification(self, 
                                     slot: str, 
                                     logits: torch.Tensor,
                                     slot_value_vocab: Dict[str, List[str]]) -> str:
        """
        Convert classification logits to actual value
        
        Args:
            slot: Slot name
            logits: [vocab_size] logits
            slot_value_vocab: Vocabulary mapping
            
        Returns:
            Predicted value string
        """
        if slot not in slot_value_vocab:
            return "dontcare"
        
        vocab = slot_value_vocab[slot]
        real_values = [v for v in vocab if v not in ['[NONE]', 'none', '']]
        
        pred_idx = torch.argmax(logits).item()
        if pred_idx < len(real_values):
            return real_values[pred_idx]
        return "dontcare"
    
    def extract_span_value(self,
                          tokens: List[str],
                          start_logits: torch.Tensor,
                          end_logits: torch.Tensor) -> str:
        """
        Extract span from tokens using start/end logits
        
        Args:
            tokens: List of tokens
            start_logits: [seq_len] start position logits
            end_logits: [seq_len] end position logits
            
        Returns:
            Extracted span text
        """
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        
        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract tokens
        span_tokens = tokens[start_idx:end_idx+1]
        return ' '.join(span_tokens)


class HybridDeltaTargetComputer:
    """
    Compute delta targets for hybrid approach
    """
    
    def __init__(self, 
                 slot_list: List[str],
                 slot_value_vocab: Dict[str, List[str]]):
        self.slot_list = slot_list
        self.slot_value_vocab = slot_value_vocab
        self.slot2idx = {slot: idx for idx, slot in enumerate(slot_list)}
        self.num_slots = len(slot_list)
        
        # Load categorization
        self.classification_slots = [
            'hotel-internet', 'hospital-department', 'hotel-parking', 
            'hotel-type', 'restaurant-pricerange', 'hotel-pricerange',
            'hotel-area', 'attraction-area', 'restaurant-area',
            'hotel-book people', 'train-day', 'hotel-stars',
            'restaurant-book people', 'hotel-book stay', 'restaurant-book day'
        ]
        
        self.span_extraction_slots = [
            'train-book people', 'hotel-book day', 'train-destination',
            'train-departure', 'attraction-type', 'restaurant-book time',
            'hotel-name', 'taxi-arriveby', 'taxi-leaveat',
            'restaurant-food', 'train-arriveby', 'attraction-name',
            'train-leaveat', 'restaurant-name', 'taxi-departure',
            'taxi-destination'
        ]
        
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
        Compute delta targets for hybrid approach
        
        Returns targets for:
        - Operations (all slots)
        - Value existence (all slots)
        - Classification values (15 slots)
        - Span positions (16 slots)
        - Special values (all slots)
        """
        
        if slot_list is None:
            slot_list = self.slot_list
        
        # Initialize labels
        operation_labels = torch.zeros(self.num_slots, dtype=torch.long)
        value_existence_labels = torch.zeros(self.num_slots)
        none_labels = torch.zeros(self.num_slots)
        dontcare_labels = torch.zeros(self.num_slots)
        
        # Classification targets (sparse - only for classification slots)
        classification_targets = {}
        
        prev_set = set(previous_belief_state.keys()) if previous_belief_state else set()
        curr_set = set(current_belief_state.keys()) if current_belief_state else set()
        
        for slot_name in self.slot_list:
            slot_idx = self.slot2idx[slot_name]
            
            prev_has_slot = slot_name in prev_set
            curr_has_slot = slot_name in curr_set
            
            # Determine operation
            if not prev_has_slot and curr_has_slot:
                operation_labels[slot_idx] = self.ADD
                value_existence_labels[slot_idx] = 1.0
                curr_value = current_belief_state[slot_name].lower()
                
                # Check special values
                if curr_value == 'none':
                    none_labels[slot_idx] = 1.0
                elif curr_value in ["dontcare", "don't care"]:
                    dontcare_labels[slot_idx] = 1.0
                
                # Classification target
                if slot_name in self.classification_slots:
                    classification_targets[slot_name] = curr_value
                    
            elif prev_has_slot and not curr_has_slot:
                operation_labels[slot_idx] = self.REMOVE
                value_existence_labels[slot_idx] = 0.0
                
            elif prev_has_slot and curr_has_slot:
                prev_value = previous_belief_state[slot_name].lower()
                curr_value = current_belief_state[slot_name].lower()
                
                if prev_value != curr_value:
                    operation_labels[slot_idx] = self.UPDATE
                    value_existence_labels[slot_idx] = 1.0
                    
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value in ["dontcare", "don't care"]:
                        dontcare_labels[slot_idx] = 1.0
                    
                    if slot_name in self.classification_slots:
                        classification_targets[slot_name] = curr_value
                else:
                    operation_labels[slot_idx] = self.KEEP
                    value_existence_labels[slot_idx] = 1.0
                    
                    if curr_value == 'none':
                        none_labels[slot_idx] = 1.0
                    elif curr_value in ["dontcare", "don't care"]:
                        dontcare_labels[slot_idx] = 1.0
            else:
                operation_labels[slot_idx] = self.KEEP
                value_existence_labels[slot_idx] = 0.0
        
        return {
            'slot_operations': operation_labels,
            'value_existence': value_existence_labels,
            'none': none_labels,
            'dontcare': dontcare_labels,
            'classification_targets': classification_targets
            # TODO: Add span targets when we have token-level annotations
        }
