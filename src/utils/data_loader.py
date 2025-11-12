#!/usr/bin/env python3
"""
Data loader utilities for DST models
Provides consistent data loading and processing interfaces
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

class DSTDataset(Dataset):
    """PyTorch Dataset for DST training/evaluation"""
    
    def __init__(self, 
                 instances: List[Dict], 
                 slot_meta: List[str],
                 max_history: int = 10,
                 include_system_response: bool = True):
        """
        Initialize DST Dataset
        
        Args:
            instances: List of dialogue instances
            slot_meta: List of slot names
            max_history: Maximum number of dialogue history turns to keep
            include_system_response: Whether to include system response in input
        """
        self.instances = instances
        self.slot_meta = slot_meta
        self.max_history = max_history
        self.include_system_response = include_system_response
        
        # Create slot to index mapping
        self.slot_to_idx = {slot: idx for idx, slot in enumerate(slot_meta)}
        self.num_slots = len(slot_meta)
        
        logger.info(f"Initialized DSTDataset with {len(instances)} instances, {self.num_slots} slots")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        instance = self.instances[idx]
        
        # Build input text
        input_text = self._build_input_text(instance)
        
        # Convert belief state to slot-value format
        belief_state_dict = self._process_belief_state(instance['belief_state'])
        
        # Create labels for each slot
        slot_labels = []
        for slot in self.slot_meta:
            value = belief_state_dict.get(slot, 'none')
            slot_labels.append(value)
        
        return {
            'dialogue_id': instance['dialogue_id'],
            'turn_id': instance['turn_id'],
            'input_text': input_text,
            'belief_state': belief_state_dict,
            'slot_labels': slot_labels,
            'domains': instance.get('domains', []),
            'is_last_turn': instance.get('is_last_turn', False)
        }
    
    def _build_input_text(self, instance: Dict) -> str:
        """Build input text from dialogue context"""
        parts = []
        
        # Add dialogue history
        if instance['dialogue_history'].strip():
            parts.append(instance['dialogue_history'])
        
        # Add system response if available and requested
        if self.include_system_response and instance.get('system_response', '').strip():
            parts.append(f"System: {instance['system_response']}")
        
        # Add current user utterance
        parts.append(f"User: {instance['user_utterance']}")
        
        return ' '.join(parts)
    
    def _process_belief_state(self, belief_state: List[List[str]]) -> Dict[str, str]:
        """Convert belief state to dictionary format"""
        state_dict = {}
        
        # Initialize with 'none' for all slots
        for slot in self.slot_meta:
            state_dict[slot] = 'none'
        
        # Fill in actual values
        for slot_value in belief_state:
            if len(slot_value) == 2:
                slot, value = slot_value
                if slot in self.slot_to_idx:
                    # Normalize value
                    if not value or value.lower() in ['none', 'not mentioned', '']:
                        state_dict[slot] = 'none'
                    elif value.lower() in ['dont care', 'dontcare', "don't care", "do not care"]:
                        state_dict[slot] = 'dontcare'
                    else:
                        state_dict[slot] = value.lower().strip()
        
        return state_dict


class DSTDataLoader:
    """Data loader manager for DST tasks"""
    
    def __init__(self, 
                 data_dir: str,
                 slot_meta_path: str,
                 batch_size: int = 16,
                 max_history: int = 10,
                 include_system_response: bool = True,
                 num_workers: int = 0):
        """
        Initialize data loader manager
        
        Args:
            data_dir: Directory containing processed data files
            slot_meta_path: Path to slot metadata file
            batch_size: Batch size for data loaders
            max_history: Maximum dialogue history length
            include_system_response: Whether to include system response
            num_workers: Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_history = max_history
        self.include_system_response = include_system_response
        self.num_workers = num_workers
        
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_data = json.load(f)
            self.slot_meta = slot_data['slot_meta']
        
        # Load datasets
        self.datasets = {}
        self.data_loaders = {}
        self._load_datasets()
        
        logger.info(f"Initialized DSTDataLoader with batch_size={batch_size}")
    
    def _load_datasets(self):
        """Load all dataset splits"""
        splits = ['train', 'dev', 'test']
        
        for split in splits:
            file_path = f"{self.data_dir}/{split}_instances.json"
            try:
                with open(file_path, 'r') as f:
                    instances = json.load(f)
                
                dataset = DSTDataset(
                    instances=instances,
                    slot_meta=self.slot_meta,
                    max_history=self.max_history,
                    include_system_response=self.include_system_response
                )
                
                self.datasets[split] = dataset
                logger.info(f"Loaded {split} dataset: {len(instances)} instances")
                
            except FileNotFoundError:
                logger.warning(f"Dataset file not found: {file_path}")
                self.datasets[split] = None
    
    def get_dataloader(self, 
                      split: str,
                      shuffle: bool = None,
                      batch_size: int = None,
                      collate_fn = None) -> DataLoader:
        """
        Get data loader for specified split
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            shuffle: Whether to shuffle data (default: True for train, False for others)
            batch_size: Batch size (default: use class default)
            collate_fn: Custom collate function
            
        Returns:
            PyTorch DataLoader
        """
        if split not in self.datasets or self.datasets[split] is None:
            raise ValueError(f"Dataset split '{split}' not available")
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        if batch_size is None:
            batch_size = self.batch_size
        
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        # Cache data loader
        cache_key = f"{split}_{shuffle}_{batch_size}"
        if cache_key not in self.data_loaders:
            self.data_loaders[cache_key] = DataLoader(
                self.datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available()
            )
        
        return self.data_loaders[cache_key]
    
    def _default_collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Default collate function for batching"""
        
        # Collect batch data
        dialogue_ids = [item['dialogue_id'] for item in batch]
        turn_ids = [item['turn_id'] for item in batch]
        input_texts = [item['input_text'] for item in batch]
        belief_states = [item['belief_state'] for item in batch]
        slot_labels = [item['slot_labels'] for item in batch]
        domains = [item['domains'] for item in batch]
        is_last_turns = [item['is_last_turn'] for item in batch]
        
        return {
            'dialogue_ids': dialogue_ids,
            'turn_ids': turn_ids,
            'input_texts': input_texts,
            'belief_states': belief_states,
            'slot_labels': slot_labels,
            'domains': domains,
            'is_last_turns': is_last_turns,
            'batch_size': len(batch)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'splits': {},
            'total_instances': 0,
            'slot_meta': self.slot_meta,
            'num_slots': len(self.slot_meta)
        }
        
        for split, dataset in self.datasets.items():
            if dataset is not None:
                split_size = len(dataset)
                stats['splits'][split] = split_size
                stats['total_instances'] += split_size
        
        # Analyze slot value distributions (using train set if available)
        if 'train' in self.datasets and self.datasets['train'] is not None:
            stats['slot_value_stats'] = self._analyze_slot_values('train')
        
        return stats
    
    def _analyze_slot_values(self, split: str) -> Dict[str, Any]:
        """Analyze slot value distributions"""
        if split not in self.datasets or self.datasets[split] is None:
            return {}
        
        slot_value_counts = {slot: {'none': 0, 'dontcare': 0, 'values': set()} 
                           for slot in self.slot_meta}
        
        dataset = self.datasets[split]
        for i in range(len(dataset)):
            item = dataset[i]
            belief_state = item['belief_state']
            
            for slot in self.slot_meta:
                value = belief_state.get(slot, 'none')
                if value == 'none':
                    slot_value_counts[slot]['none'] += 1
                elif value == 'dontcare':
                    slot_value_counts[slot]['dontcare'] += 1
                else:
                    slot_value_counts[slot]['values'].add(value)
        
        # Convert sets to counts and lists
        slot_stats = {}
        for slot, counts in slot_value_counts.items():
            slot_stats[slot] = {
                'none_count': counts['none'],
                'dontcare_count': counts['dontcare'],
                'unique_values': len(counts['values']),
                'sample_values': list(counts['values'])[:10]  # First 10 values as examples
            }
        
        return slot_stats


class TokenizedDSTDataset(DSTDataset):
    """Tokenized version of DST Dataset for transformer models"""
    
    def __init__(self,
                 instances: List[Dict],
                 slot_meta: List[str],
                 tokenizer,
                 max_seq_length: int = 512,
                 max_history: int = 10,
                 include_system_response: bool = True):
        """
        Initialize tokenized DST dataset
        
        Args:
            instances: List of dialogue instances
            slot_meta: List of slot names
            tokenizer: Hugging Face tokenizer
            max_seq_length: Maximum sequence length
            max_history: Maximum dialogue history length
            include_system_response: Whether to include system response
        """
        super().__init__(instances, slot_meta, max_history, include_system_response)
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        logger.info(f"Initialized TokenizedDSTDataset with max_seq_length={max_seq_length}")
    
    def __getitem__(self, idx):
        """Get tokenized dataset item"""
        # Get base item
        item = super().__getitem__(idx)
        
        # Tokenize input text
        encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Add tokenized fields
        item['input_ids'] = encoding['input_ids'].squeeze(0)
        item['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        return item


def create_tokenized_data_loader(data_dir: str,
                                slot_meta_path: str,
                                tokenizer,
                                batch_size: int = 16,
                                max_seq_length: int = 512,
                                max_history: int = 10,
                                num_workers: int = 0) -> DSTDataLoader:
    """
    Create a data loader with tokenized datasets
    
    Args:
        data_dir: Directory containing processed data
        slot_meta_path: Path to slot metadata
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        max_history: Maximum dialogue history length
        num_workers: Number of data loading workers
        
    Returns:
        DSTDataLoader with tokenized datasets
    """
    
    # Load slot metadata
    with open(slot_meta_path, 'r') as f:
        slot_data = json.load(f)
        slot_meta = slot_data['slot_meta']
    
    class TokenizedDSTDataLoader(DSTDataLoader):
        def _load_datasets(self):
            """Load tokenized datasets"""
            splits = ['train', 'dev', 'test']
            
            for split in splits:
                file_path = f"{self.data_dir}/{split}_instances.json"
                try:
                    with open(file_path, 'r') as f:
                        instances = json.load(f)
                    
                    dataset = TokenizedDSTDataset(
                        instances=instances,
                        slot_meta=self.slot_meta,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        max_history=self.max_history,
                        include_system_response=self.include_system_response
                    )
                    
                    self.datasets[split] = dataset
                    logger.info(f"Loaded tokenized {split} dataset: {len(instances)} instances")
                    
                except FileNotFoundError:
                    logger.warning(f"Dataset file not found: {file_path}")
                    self.datasets[split] = None
        
        def _default_collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
            """Collate function for tokenized data"""
            base_batch = super()._default_collate_fn(batch)
            
            # Stack tokenized tensors
            if 'input_ids' in batch[0]:
                base_batch['input_ids'] = torch.stack([item['input_ids'] for item in batch])
                base_batch['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
                
                if 'token_type_ids' in batch[0]:
                    base_batch['token_type_ids'] = torch.stack([item['token_type_ids'] for item in batch])
            
            return base_batch
    
    return TokenizedDSTDataLoader(
        data_dir=data_dir,
        slot_meta_path=slot_meta_path,
        batch_size=batch_size,
        max_history=max_history,
        num_workers=num_workers
    )


def main():
    """Example usage of data loader utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DST data loader")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--slot_meta", type=str, default="data/processed/slot_meta.json",
                        help="Path to slot metadata file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for testing")
    
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = DSTDataLoader(
        data_dir=args.data_dir,
        slot_meta_path=args.slot_meta,
        batch_size=args.batch_size
    )
    
    # Print statistics
    stats = data_loader.get_statistics()
    print("Dataset Statistics:")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Number of slots: {stats['num_slots']}")
    for split, size in stats['splits'].items():
        print(f"{split}: {size} instances")
    
    # Test data loading
    for split in ['train', 'dev', 'test']:
        if split in data_loader.datasets and data_loader.datasets[split] is not None:
            dataloader = data_loader.get_dataloader(split)
            
            print(f"\nTesting {split} data loader:")
            for i, batch in enumerate(dataloader):
                print(f"Batch {i+1}:")
                print(f"  Batch size: {batch['batch_size']}")
                print(f"  Sample dialogue ID: {batch['dialogue_ids'][0]}")
                print(f"  Sample input length: {len(batch['input_texts'][0].split())}")
                
                if i >= 2:  # Only show first 3 batches
                    break


if __name__ == "__main__":
    main()