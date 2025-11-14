"""
DataLoader for Multi-Level GNN-based DST with Delta Prediction
Loads structured dialogue_history for graph construction
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple
import os


class GraphDSTDataset(Dataset):
    """
    Dataset for GNN-based DST with delta prediction
    
    Each instance contains:
        - user_utterance: str
        - dialogue_history: List[Dict] (structured turns for graph building)
        - previous_belief_state: Dict[str, str]
        - delta_belief_state: Dict[str, str] (target - only changes)
        - current_belief_state: Dict[str, str] (for evaluation)
    """
    
    def __init__(self, data_path: str, slot_meta_path: str):
        """
        Args:
            data_path: Path to instances JSON file
            slot_meta_path: Path to slot_meta.json
        """
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loading slot metadata from {slot_meta_path}...")
        with open(slot_meta_path, 'r') as f:
            self.slot_meta = json.load(f)
        
        # Create slot to index mapping
        self.slot2idx = {slot: idx for idx, slot in enumerate(self.slot_meta)}
        self.idx2slot = {idx: slot for slot, idx in self.slot2idx.items()}
        
        # Value vocabulary (will be built from data)
        self.value2idx = {'[PAD]': 0, '[UNK]': 1, 'none': 2, 'dontcare': 3}
        self.idx2value = {0: '[PAD]', 1: '[UNK]', 2: 'none', 3: 'dontcare'}
        self.slot_value_vocab = {slot: set() for slot in self.slot_meta}
        self._build_value_vocab()
        
        print(f"✓ Loaded {len(self.data)} instances")
        print(f"✓ Slot vocabulary: {len(self.slot_meta)} slots")
        print(f"✓ Value vocabulary: {len(self.value2idx)} values")
    
    def _build_value_vocab(self):
        """Build value vocabulary from all belief states"""
        values = set()
        
        for inst in self.data:
            # Collect from current, delta, and previous belief states
            for state in [
                inst.get('current_belief_state', {}),
                inst.get('delta_belief_state', {}),
                inst.get('previous_belief_state', {})
            ]:
                for slot, value in state.items():
                    if value is None:
                        continue
                    norm_value = str(value).strip()
                    if not norm_value:
                        continue
                    values.add(norm_value)
                    self.slot_value_vocab.setdefault(slot, set()).add(norm_value)
            
            # Collect from dialogue history
            for turn in inst.get('dialogue_history', []):
                belief_state = turn.get('belief_state', {})
                if isinstance(belief_state, dict):
                    for slot, value in belief_state.items():
                        if value is None:
                            continue
                        norm_value = str(value).strip()
                        if not norm_value:
                            continue
                        values.add(norm_value)
                        self.slot_value_vocab.setdefault(slot, set()).add(norm_value)

        # Ensure special tokens present per slot
        for slot, slot_values in self.slot_value_vocab.items():
            slot_values.update({'none', 'dontcare'})

        # Add to vocabulary
        for value in sorted(values):
            if value not in self.value2idx:
                idx = len(self.value2idx)
                self.value2idx[value] = idx
                self.idx2value[idx] = value

        # Finalise per-slot vocab as sorted lists
        self.slot_value_vocab = {
            slot: sorted(v for v in values if v)
            for slot, values in self.slot_value_vocab.items()
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns instance dict with all fields intact
        Graph construction will happen in collate_fn or model
        """
        inst = self.data[idx]
        
        return {
            'dialogue_id': inst['dialogue_id'],
            'turn_id': inst['turn_id'],
            'user_utterance': inst['user_utterance'],
            'system_response': inst.get('system_response', ''),
            'dialogue_history': inst.get('dialogue_history', []),  # List[Dict]
            'previous_belief_state': inst.get('previous_belief_state', {}),
            'current_belief_state': inst.get('current_belief_state', {}),
            'delta_belief_state': inst.get('delta_belief_state', {}),
            'domains': inst.get('domains', []),
            'is_last_turn': inst.get('is_last_turn', False)
        }


def collate_fn_graph(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that preserves structured data for graph construction
    
    Unlike traditional collate that converts to tensors,
    this keeps dialogue_history as List[Dict] for MultiLevelGraphBuilder
    
    Returns:
        batch_dict with:
            - utterances: List[str]
            - dialogue_histories: List[List[Dict]]
            - previous_belief_states: List[Dict]
            - delta_belief_states: List[Dict]
            - current_belief_states: List[Dict]
            - metadata: dialogue_id, turn_id, etc.
    """
    return {
        'dialogue_ids': [item['dialogue_id'] for item in batch],
        'turn_ids': [item['turn_id'] for item in batch],
        'utterances': [item['user_utterance'] for item in batch],
        'system_responses': [item['system_response'] for item in batch],
        'dialogue_histories': [item['dialogue_history'] for item in batch],
        'previous_belief_states': [item['previous_belief_state'] for item in batch],
        'current_belief_states': [item['current_belief_state'] for item in batch],
        'delta_belief_states': [item['delta_belief_state'] for item in batch],
        'domains': [item['domains'] for item in batch],
        'is_last_turn': [item['is_last_turn'] for item in batch],
        'batch_size': len(batch)
    }


def create_dataloaders(data_dir: str, 
                       batch_size: int = 8,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, GraphDSTDataset]:
    """
    Create train/dev/test dataloaders
    
    Returns:
        train_loader, dev_loader, test_loader, train_dataset
    """
    slot_meta_path = os.path.join(data_dir, 'slot_meta.json')
    
    # Create datasets
    train_dataset = GraphDSTDataset(
        data_path=os.path.join(data_dir, 'train_instances.json'),
        slot_meta_path=slot_meta_path
    )
    
    dev_dataset = GraphDSTDataset(
        data_path=os.path.join(data_dir, 'dev_instances.json'),
        slot_meta_path=slot_meta_path
    )
    
    test_dataset = GraphDSTDataset(
        data_path=os.path.join(data_dir, 'test_instances.json'),
        slot_meta_path=slot_meta_path
    )
    
    # Share vocabularies
    dev_dataset.slot2idx = train_dataset.slot2idx
    dev_dataset.idx2slot = train_dataset.idx2slot
    dev_dataset.value2idx = train_dataset.value2idx
    dev_dataset.idx2value = train_dataset.idx2value
    dev_dataset.slot_value_vocab = train_dataset.slot_value_vocab
    
    test_dataset.slot2idx = train_dataset.slot2idx
    test_dataset.idx2slot = train_dataset.idx2slot
    test_dataset.value2idx = train_dataset.value2idx
    test_dataset.idx2value = train_dataset.idx2value
    test_dataset.slot_value_vocab = train_dataset.slot_value_vocab
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_graph,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_graph,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_graph,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print(f"DATALOADER SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_dataset)} instances, {len(train_loader)} batches")
    print(f"Dev:   {len(dev_dataset)} instances, {len(dev_loader)} batches")
    print(f"Test:  {len(test_dataset)} instances, {len(test_loader)} batches")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    return train_loader, dev_loader, test_loader, train_dataset


if __name__ == "__main__":
    # Test dataloader
    print("Testing GraphDSTDataset and DataLoader...\n")
    
    train_loader, dev_loader, test_loader, train_dataset = create_dataloaders(
        data_dir='data/processed_graph',
        batch_size=4
    )
    
    # Test one batch
    print("\n=== Testing one batch ===")
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch size: {batch['batch_size']}")
    print(f"\nFirst instance:")
    print(f"  Dialogue ID: {batch['dialogue_ids'][0]}")
    print(f"  Turn ID: {batch['turn_ids'][0]}")
    print(f"  Utterance: {batch['utterances'][0][:80]}")
    print(f"  Dialogue history: List[Dict] with {len(batch['dialogue_histories'][0])} turns")
    if len(batch['dialogue_histories'][0]) > 0:
        print(f"    First turn keys: {batch['dialogue_histories'][0][0].keys()}")
    print(f"  Previous belief: {len(batch['previous_belief_states'][0])} slots")
    print(f"  Delta belief: {len(batch['delta_belief_states'][0])} slots")
    print(f"    Delta: {batch['delta_belief_states'][0]}")
    
    print("\n✓ DataLoader test passed!")
