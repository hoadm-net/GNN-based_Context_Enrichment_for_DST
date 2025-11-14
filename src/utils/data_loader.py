"""
Data Loader for History-Aware GraphDST

Handles loading và preprocessing của MultiWOZ 2.4 data cho training và validation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from transformers import BertTokenizer
import random


class DSTDataset(Dataset):
    """
    PyTorch Dataset cho Dialog State Tracking
    
    Features:
    - Configurable dialogue history length
    - BERT tokenization cho current utterance
    - Belief state normalization
    - Graph construction data preparation
    """
    
    def __init__(self, 
                 data_path: str,
                 slot_meta_path: str,
                 tokenizer: BertTokenizer,
                 max_sequence_length: int = 512,
                 max_history_length: int = 10,
                 include_history: bool = True):
        
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_history_length = max_history_length
        self.include_history = include_history
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Group data by dialogue and compute delta targets
        self.data = self._prepare_delta_targets(raw_data)
        
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_meta = json.load(f)
            # Handle different formats of slot_meta.json
            if isinstance(slot_meta.get('slot_meta'), list):
                # Format: {"slot_meta": [...]}
                self.slot_list = slot_meta['slot_meta']
                self.slot_meta = {slot: {'type': 'categorical'} for slot in self.slot_list}
            else:
                # Format: {"slot_meta": {...}, "slot_list": [...]}
                self.slot_meta = slot_meta.get('slot_meta', {})
                self.slot_list = slot_meta.get('slot_list', list(self.slot_meta.keys()))
            
            self.domain_list = slot_meta.get('domain_list', ['hotel', 'restaurant', 'attraction', 'train', 'taxi'])
        
        # Create mappings
        self.slot2idx = {slot: idx for idx, slot in enumerate(self.slot_list)}
        self.domain2idx = {domain: idx for idx, domain in enumerate(self.domain_list)}
        
        print(f"Loaded {len(self.data)} instances from {data_path}")
        print(f"Slot vocabulary: {len(self.slot_list)} slots")
        print(f"Domains: {self.domain_list}")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training instance
        
        Returns:
            Dictionary containing:
            - input_ids, attention_mask: Tokenized current utterance
            - history_data: Previous turns và belief states
            - current_belief_state: Target belief state
            - domain_labels: Active domains
            - slot_labels: Active slots
            - dialogue_id, turn_id: Identifiers
        """
        instance = self.data[idx]
        
        # Extract basic information
        dialogue_id = instance['dialogue_id']
        turn_id = instance['turn_id']
        user_utterance = instance['user_utterance']
        system_response = instance.get('system_response', '')
        dialogue_history = instance.get('dialogue_history', [])
        belief_state = instance['belief_state']
        domains = instance.get('domains', [])
        
        # Delta information (added by _prepare_delta_targets)
        prev_belief_state = instance.get('prev_belief_state', {})
        curr_belief_state = instance.get('curr_belief_state', {})
        
        # Tokenize current utterance
        current_text = user_utterance
        if system_response:
            current_text = f"[USER] {user_utterance} [SYSTEM] {system_response}"
            
        encoding = self.tokenizer(
            current_text,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Process belief state with delta targets
        belief_state_dict, domain_labels, delta_targets = self._process_belief_state(prev_belief_state, curr_belief_state, domains)
        
        # Process history (if enabled)
        history_data = []
        if self.include_history and dialogue_history:
            # Check if dialogue_history is string or list
            if isinstance(dialogue_history, str):
                # If it's a string, we'll create empty history for now
                history_data = []
            else:
                history_data = self._process_dialogue_history(dialogue_history)
        
        return {
            # Current utterance (for BERT)
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'utterance_text': current_text,
            
            # History (for GNN)
            'history_data': history_data,
            
            # Labels for delta DST
            'belief_state': belief_state_dict,
            'prev_belief_state': prev_belief_state,
            'curr_belief_state': curr_belief_state,
            'domain_labels': torch.tensor(domain_labels, dtype=torch.float),
            
            # Delta targets
            'operation_labels': torch.tensor(delta_targets['operation_labels'], dtype=torch.long),
            'value_existence_labels': torch.tensor(delta_targets['value_existence_labels'], dtype=torch.float),
            'none_labels': torch.tensor(delta_targets['none_labels'], dtype=torch.float),
            'dontcare_labels': torch.tensor(delta_targets['dontcare_labels'], dtype=torch.float),
            
            # Metadata
            'dialogue_id': dialogue_id,
            'turn_id': turn_id,
            'domains': domains,
            'is_last_turn': instance.get('is_last_turn', False)
        }
    
    def _process_belief_state(self, prev_belief_state: Dict[str, str], curr_belief_state: Dict[str, str], domains: List[str]) -> Tuple[Dict, List[float], Dict]:
        """
        Process belief state into delta format
        
        Args:
            prev_belief_state: Previous turn's belief state
            curr_belief_state: Current turn's belief state
            domains: List of active domains
            
        Returns:
            curr_belief_state_dict: Current belief state
            domain_labels: Binary labels cho domains  
            delta_targets: Delta training targets
        """
        # Initialize domain labels
        domain_labels = [0.0] * len(self.domain_list)
        
        # Set active domain labels
        for domain in domains:
            if domain in self.domain2idx:
                domain_labels[self.domain2idx[domain]] = 1.0
        
        # Compute delta targets using DeltaTargetComputer logic
        delta_targets = self._compute_delta_targets(prev_belief_state, curr_belief_state)
        
        return curr_belief_state, domain_labels, delta_targets
    
    def _compute_delta_targets(self, prev_belief_state: Dict[str, str], curr_belief_state: Dict[str, str]) -> Dict:
        """
        Compute delta targets between consecutive belief states
        """
        # Operation indices
        KEEP, ADD, UPDATE, REMOVE = 0, 1, 2, 3
        
        # Initialize targets
        operation_labels = [KEEP] * len(self.slot_list)
        value_existence_labels = [0.0] * len(self.slot_list)
        none_labels = [0.0] * len(self.slot_list)
        dontcare_labels = [0.0] * len(self.slot_list)
        
        prev_set = set(prev_belief_state.keys())
        curr_set = set(curr_belief_state.keys())
        
        for slot_name in self.slot_list:
            slot_idx = self.slot2idx[slot_name]
            
            prev_has_slot = slot_name in prev_set
            curr_has_slot = slot_name in curr_set
            
            if not prev_has_slot and curr_has_slot:
                # ADD operation
                operation_labels[slot_idx] = ADD
                value_existence_labels[slot_idx] = 1.0
                
                # Check for special values
                curr_value = curr_belief_state[slot_name].lower()
                if curr_value == 'none':
                    none_labels[slot_idx] = 1.0
                elif curr_value in ["dontcare", "don't care"]:
                    dontcare_labels[slot_idx] = 1.0
                    
            elif prev_has_slot and not curr_has_slot:
                # REMOVE operation  
                operation_labels[slot_idx] = REMOVE
                value_existence_labels[slot_idx] = 0.0
                
            elif prev_has_slot and curr_has_slot:
                prev_value = prev_belief_state[slot_name].lower()
                curr_value = curr_belief_state[slot_name].lower()
                
                if prev_value != curr_value:
                    # UPDATE operation
                    operation_labels[slot_idx] = UPDATE
                else:
                    # KEEP operation (no change)
                    operation_labels[slot_idx] = KEEP
                    
                # Set value existence and special value labels
                value_existence_labels[slot_idx] = 1.0
                if curr_value == 'none':
                    none_labels[slot_idx] = 1.0
                elif curr_value in ["dontcare", "don't care"]:
                    dontcare_labels[slot_idx] = 1.0
            else:
                # Both empty - KEEP with no value
                operation_labels[slot_idx] = KEEP
                value_existence_labels[slot_idx] = 0.0
                
        return {
            'operation_labels': operation_labels,
            'value_existence_labels': value_existence_labels,
            'none_labels': none_labels,
            'dontcare_labels': dontcare_labels
        }
    
    def _prepare_delta_targets(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Group data by dialogue and compute delta targets
        
        Args:
            raw_data: List of raw instances
            
        Returns:
            List of instances with delta targets
        """
        # Group by dialogue_id
        dialogues = {}
        for instance in raw_data:
            dialogue_id = instance['dialogue_id']
            if dialogue_id not in dialogues:
                dialogues[dialogue_id] = []
            dialogues[dialogue_id].append(instance)
        
        # Sort turns within each dialogue
        for dialogue_id in dialogues:
            dialogues[dialogue_id].sort(key=lambda x: x['turn_id'])
        
        # Compute delta targets
        processed_data = []
        for dialogue_id, turns in dialogues.items():
            for i, turn in enumerate(turns):
                # Get previous belief state
                if i == 0:
                    prev_belief_state = {}  # Empty for first turn
                else:
                    prev_turn = turns[i-1]
                    prev_belief_state = {slot: value for slot, value in prev_turn['belief_state']}
                
                # Current belief state
                curr_belief_state = {slot: value for slot, value in turn['belief_state']}
                
                # Add delta information to instance
                turn_copy = turn.copy()
                turn_copy['prev_belief_state'] = prev_belief_state
                turn_copy['curr_belief_state'] = curr_belief_state
                
                processed_data.append(turn_copy)
        
        return processed_data
    
    def _process_dialogue_history(self, history: List[Dict]) -> List[Dict]:
        """
        Process dialogue history for graph construction
        
        Args:
            history: List of previous turns
            
        Returns:
            Processed history data
        """
        processed_history = []
        
        # Take last N turns
        recent_history = history[-self.max_history_length:] if len(history) > self.max_history_length else history
        
        for turn_data in recent_history:
            processed_turn = {
                'turn_id': turn_data.get('turn_id', 0),
                'user_utterance': turn_data.get('user_utterance', ''),
                'system_response': turn_data.get('system_response', ''),
                'belief_state': turn_data.get('belief_state', []),
                'domains': turn_data.get('domains', [])
            }
            processed_history.append(processed_turn)
        
        return processed_history


class DSTDataLoader:
    """
    Data loader wrapper với advanced features
    """
    
    def __init__(self,
                 data_dir: str,
                 slot_meta_path: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 tokenizer_name: str = "bert-base-uncased",
                 max_sequence_length: int = 512,
                 max_history_length: int = 10):
        
        self.data_dir = data_dir
        self.slot_meta_path = slot_meta_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.max_history_length = max_history_length
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        print(f"Initialized DSTDataLoader with tokenizer: {tokenizer_name}")
    
    def get_dataloader(self, split: str, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader cho specified split
        
        Args:
            split: 'train', 'dev', hoặc 'test'
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        # Determine data file path
        if split == 'train':
            data_path = os.path.join(self.data_dir, 'train_instances.json')
        elif split == 'dev' or split == 'val':
            data_path = os.path.join(self.data_dir, 'dev_instances.json')
        elif split == 'test':
            data_path = os.path.join(self.data_dir, 'test_instances.json')
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Create dataset
        dataset = DSTDataset(
            data_path=data_path,
            slot_meta_path=self.slot_meta_path,
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            max_history_length=self.max_history_length
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        
        return dataloader
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function để handle variable-length data
        """
        # Stack tensor fields
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        domain_labels = torch.stack([item['domain_labels'] for item in batch])
        
        # Stack delta target tensors
        operation_labels = torch.stack([item['operation_labels'] for item in batch])
        value_existence_labels = torch.stack([item['value_existence_labels'] for item in batch])
        none_labels = torch.stack([item['none_labels'] for item in batch])
        dontcare_labels = torch.stack([item['dontcare_labels'] for item in batch])
        
        # Collect other fields
        batch_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'domain_labels': domain_labels,
            
            # Delta targets
            'operation_labels': operation_labels,
            'value_existence_labels': value_existence_labels,
            'none_labels': none_labels,
            'dontcare_labels': dontcare_labels,
            
            # Lists of data
            'utterance_texts': [item['utterance_text'] for item in batch],
            'history_data': [item['history_data'] for item in batch],
            'belief_states': [item['belief_state'] for item in batch],
            'prev_belief_states': [item['prev_belief_state'] for item in batch],
            'curr_belief_states': [item['curr_belief_state'] for item in batch],
            'dialogue_ids': [item['dialogue_id'] for item in batch],
            'turn_ids': [item['turn_id'] for item in batch],
            'domains': [item['domains'] for item in batch],
            'is_last_turns': [item['is_last_turn'] for item in batch]
        }
        
        return batch_data


if __name__ == "__main__":
    # Test data loader
    data_loader = DSTDataLoader(
        data_dir="data/processed",
        slot_meta_path="data/processed/slot_meta.json",
        batch_size=4,
        num_workers=0  # No multiprocessing for testing
    )
    
    try:
        # Test loading
        train_loader = data_loader.get_dataloader('train', shuffle=False)
        print(f"Train loader created with {len(train_loader)} batches")
        
        # Test one batch
        for batch in train_loader:
            print("\nBatch contents:")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Domain labels shape: {batch['domain_labels'].shape}")
            print(f"Slot labels shape: {batch['slot_labels'].shape}")
            print(f"Number of utterances: {len(batch['utterance_texts'])}")
            print(f"Number of histories: {len(batch['history_data'])}")
            break
            
    except Exception as e:
        print(f"Error testing data loader: {e}")
        print("This is expected if processed data files don't exist yet.")