"""
GraphDST Data Loader

This module provides data loading utilities specifically designed for the GraphDST model,
building on top of the existing data pipeline but adding graph-specific features.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict


class GraphDSTDataset(Dataset):
    """Dataset for GraphDST model training"""
    
    def __init__(self, data_path: str, slot_meta_path: str, tokenizer, 
                 max_length: int = 512, history_turns: int = 3):
        """
        Initialize GraphDST Dataset
        
        Args:
            data_path: Path to processed instances JSON file
            slot_meta_path: Path to slot metadata JSON
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            history_turns: Number of dialog history turns to include
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.history_turns = history_turns
        
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_data = json.load(f)
            self.slot_names = slot_data['slot_meta']
            self.num_slots = len(self.slot_names)
        
        # Create slot name to index mapping
        self.slot_to_idx = {slot: idx for idx, slot in enumerate(self.slot_names)}
        
        # Create domain mapping
        self.domain_names = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domain_names)}
        
        # Load and process data
        with open(data_path, 'r') as f:
            self.instances = json.load(f)
        
        print(f"Loaded {len(self.instances)} instances from {data_path}")
        print(f"Found {self.num_slots} slots: {self.slot_names[:5]}..." + ("" if self.num_slots <= 5 else f" and {self.num_slots - 5} more"))
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        # Build input text (dialog history + current utterance)
        input_text = self._build_input_text(instance)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract belief state information
        belief_state = instance['belief_state']
        domains = set(instance['domains'])
        
        # Create labels
        labels = self._create_labels(belief_state, domains, encoding['input_ids'].squeeze())
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'dialogue_id': instance['dialogue_id'],
            'turn_id': instance['turn_id'],
            'input_text': input_text  # For debugging
        }
    
    def _build_input_text(self, instance: Dict) -> str:
        """Build input text with dialog history and current utterance"""
        parts = []
        
        # Add dialog history if available
        if instance['dialogue_history'].strip():
            parts.append(f"[HISTORY] {instance['dialogue_history']}")
        
        # Add current user utterance
        if instance['user_utterance'].strip():
            parts.append(f"[USER] {instance['user_utterance']}")
        
        # Add system response if available
        if instance['system_response'].strip():
            parts.append(f"[SYS] {instance['system_response']}")
        
        return " ".join(parts)
    
    def _create_labels(self, belief_state: List[List[str]], domains: set, 
                      input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create labels for multi-task learning"""
        labels = {}
        
        # 1. Domain labels (multi-label binary)
        domain_labels = torch.zeros(len(self.domain_names), dtype=torch.float)
        for domain in domains:
            if domain in self.domain_to_idx:
                domain_labels[self.domain_to_idx[domain]] = 1.0
        labels['domain_labels'] = domain_labels
        
        # 2. Slot activation labels
        active_slots = set()
        slot_values = {}
        
        for slot_value_pair in belief_state:
            if len(slot_value_pair) >= 2:
                slot_name = slot_value_pair[0]
                slot_value = slot_value_pair[1]
                
                if slot_name in self.slot_to_idx:
                    active_slots.add(slot_name)
                    slot_values[slot_name] = slot_value
        
        # Create binary labels for each slot
        for slot_name in self.slot_names:
            label_key = f"{slot_name}_active"
            labels[label_key] = torch.tensor(1 if slot_name in active_slots else 0, dtype=torch.long)
        
        # 3. Value span labels (simplified - just mark as -1 for now)
        seq_len = input_ids.size(0)
        labels['span_start_labels'] = torch.full((seq_len,), -1, dtype=torch.long)
        labels['span_end_labels'] = torch.full((seq_len,), -1, dtype=torch.long)
        
        # Try to find value spans in the input text (basic approach)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for slot_name, slot_value in slot_values.items():
            if slot_value and slot_value.lower() != 'none':
                # Simple token matching (can be improved)
                value_tokens = self.tokenizer.tokenize(slot_value.lower())
                if value_tokens:
                    # Find first occurrence
                    for i in range(len(input_tokens) - len(value_tokens) + 1):
                        if input_tokens[i:i+len(value_tokens)] == value_tokens:
                            labels['span_start_labels'][i] = 1  # Mark start
                            labels['span_end_labels'][i+len(value_tokens)-1] = 1  # Mark end
                            break
        
        return labels


class GraphDSTDataLoader:
    """Data loader for GraphDST model with custom collate function"""
    
    def __init__(self, data_dir: str, slot_meta_path: str, tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512, history_turns: int = 3):
        """
        Initialize GraphDST Data Loader
        
        Args:
            data_dir: Directory containing processed data files
            slot_meta_path: Path to slot metadata
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            history_turns: Number of dialog history turns
        """
        self.data_dir = data_dir
        self.slot_meta_path = slot_meta_path
        self.max_length = max_length
        self.history_turns = history_turns
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load datasets
        self.datasets = {}
        for split in ['train', 'dev', 'test']:
            data_path = f"{data_dir}/{split}_instances.json"
            try:
                dataset = GraphDSTDataset(
                    data_path=data_path,
                    slot_meta_path=slot_meta_path,
                    tokenizer=self.tokenizer,
                    max_length=max_length,
                    history_turns=history_turns
                )
                self.datasets[split] = dataset
                print(f"✅ Loaded {split} dataset: {len(dataset)} instances")
            except FileNotFoundError:
                print(f"⚠️  {split} dataset not found at {data_path}")
    
    def get_dataloader(self, split: str, batch_size: int = 16, shuffle: bool = False,
                      num_workers: int = 0) -> DataLoader:
        """Get DataLoader for specified split"""
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available. Available splits: {list(self.datasets.keys())}")
        
        return DataLoader(
            self.datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect labels
        labels = {}
        
        # Domain labels
        domain_labels = torch.stack([item['labels']['domain_labels'] for item in batch])
        labels['domain_labels'] = domain_labels
        
        # Slot activation labels
        if batch:
            slot_names = self.datasets[list(self.datasets.keys())[0]].slot_names
            for slot_name in slot_names:
                label_key = f"{slot_name}_active"
                slot_labels = torch.stack([item['labels'][label_key] for item in batch])
                labels[label_key] = slot_labels
        
        # Span labels
        span_start_labels = torch.stack([item['labels']['span_start_labels'] for item in batch])
        span_end_labels = torch.stack([item['labels']['span_end_labels'] for item in batch])
        labels['span_start_labels'] = span_start_labels
        labels['span_end_labels'] = span_end_labels
        
        # Metadata
        dialogue_ids = [item['dialogue_id'] for item in batch]
        turn_ids = [item['turn_id'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'dialogue_ids': dialogue_ids,
            'turn_ids': turn_ids
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {}
        
        for split, dataset in self.datasets.items():
            # Analyze domains and slots
            domain_counts = defaultdict(int)
            slot_counts = defaultdict(int)
            value_lengths = []
            
            for instance in dataset.instances:
                # Count domains
                for domain in instance['domains']:
                    domain_counts[domain] += 1
                
                # Count slots and values
                for slot_value_pair in instance['belief_state']:
                    if len(slot_value_pair) >= 2:
                        slot_name = slot_value_pair[0]
                        slot_value = slot_value_pair[1]
                        
                        slot_counts[slot_name] += 1
                        if slot_value and slot_value.lower() != 'none':
                            value_lengths.append(len(slot_value.split()))
            
            stats[split] = {
                'num_instances': len(dataset),
                'domain_distribution': dict(domain_counts),
                'slot_distribution': dict(sorted(slot_counts.items())),
                'avg_value_length': np.mean(value_lengths) if value_lengths else 0,
                'max_value_length': max(value_lengths) if value_lengths else 0
            }
        
        return stats


def create_graphdst_dataloaders(data_dir: str, slot_meta_path: str, 
                               tokenizer_name: str = "bert-base-uncased",
                               batch_size: int = 16, max_length: int = 512) -> Dict[str, DataLoader]:
    """
    Create GraphDST dataloaders for all splits
    
    Args:
        data_dir: Directory with processed data
        slot_meta_path: Path to slot metadata
        tokenizer_name: Tokenizer name
        batch_size: Batch size
        max_length: Max sequence length
    
    Returns:
        Dictionary of dataloaders
    """
    # Initialize data loader
    graphdst_loader = GraphDSTDataLoader(
        data_dir=data_dir,
        slot_meta_path=slot_meta_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # Create dataloaders
    dataloaders = {}
    
    for split in ['train', 'dev', 'test']:
        if split in graphdst_loader.datasets:
            dataloaders[split] = graphdst_loader.get_dataloader(
                split=split,
                batch_size=batch_size,
                shuffle=(split == 'train'),  # Only shuffle training data
                num_workers=0  # Set to 0 for compatibility
            )
    
    return dataloaders


if __name__ == "__main__":
    print("=" * 60)
    print("GraphDST Data Loader Test")
    print("=" * 60)
    
    # Test data loading
    data_dir = "data/processed"
    slot_meta_path = "data/processed/slot_meta.json"
    
    # Create data loaders
    dataloaders = create_graphdst_dataloaders(
        data_dir=data_dir,
        slot_meta_path=slot_meta_path,
        batch_size=4,
        max_length=256
    )
    
    print(f"Created dataloaders for: {list(dataloaders.keys())}")
    
    # Test a batch
    if 'train' in dataloaders:
        print("\n" + "=" * 40)
        print("Testing train dataloader...")
        
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Domain labels shape: {batch['labels']['domain_labels'].shape}")
        
        # Show first slot activation label
        slot_keys = [k for k in batch['labels'].keys() if k.endswith('_active')]
        if slot_keys:
            first_slot = slot_keys[0]
            print(f"Sample slot label ({first_slot}): {batch['labels'][first_slot].shape}")
        
        print(f"Span start labels shape: {batch['labels']['span_start_labels'].shape}")
        print(f"Dialogue IDs: {batch['dialogue_ids'][:2]}...")
        
        print("✅ Data loading test successful!")
    
    else:
        print("⚠️  No train dataloader available for testing")