"""
Slot Vocabulary Builder

This module builds slot vocabularies from the training data for categorical value prediction.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple
import os


def build_slot_vocabularies(data_path: str, min_freq: int = 1) -> Dict[str, List[str]]:
    """
    Build vocabulary for each slot from training data
    
    Args:
        data_path: Path to training instances JSON file
        min_freq: Minimum frequency for a value to be included
    
    Returns:
        Dictionary mapping slot names to their vocabularies
    """
    print(f"Building slot vocabularies from {data_path}...")
    
    # Load training data
    with open(data_path, 'r') as f:
        instances = json.load(f)
    
    # Count values for each slot
    slot_value_counts = defaultdict(lambda: defaultdict(int))
    
    for instance in instances:
        for slot_value_pair in instance['belief_state']:
            if len(slot_value_pair) >= 2:
                slot_name = slot_value_pair[0]
                slot_value = slot_value_pair[1]
                
                if slot_value and slot_value.lower() != 'none':
                    slot_value_counts[slot_name][slot_value] += 1
    
    # Build vocabularies
    slot_vocabularies = {}
    
    for slot_name, value_counts in slot_value_counts.items():
        # Filter by minimum frequency and sort by frequency
        vocab = []
        for value, count in value_counts.items():
            if count >= min_freq:
                vocab.append((value, count))
        
        # Sort by frequency (descending) then alphabetically
        vocab.sort(key=lambda x: (-x[1], x[0]))
        
        # Extract just the values and add special tokens
        slot_vocab = ['[NONE]'] + [item[0] for item in vocab]
        slot_vocabularies[slot_name] = slot_vocab
        
        print(f"  {slot_name}: {len(slot_vocab)} values ({len([v for v in vocab if v[1] >= 5])} frequent)")
    
    return slot_vocabularies


def save_slot_vocabularies(vocabularies: Dict[str, List[str]], output_path: str):
    """Save slot vocabularies to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(vocabularies, f, indent=2)
    print(f"Saved slot vocabularies to {output_path}")


def load_slot_vocabularies(vocab_path: str) -> Dict[str, List[str]]:
    """Load slot vocabularies from JSON file"""
    with open(vocab_path, 'r') as f:
        vocabularies = json.load(f)
    return vocabularies


def get_vocabulary_statistics(vocabularies: Dict[str, List[str]]) -> Dict:
    """Get statistics about slot vocabularies"""
    stats = {
        'total_slots': len(vocabularies),
        'total_values': sum(len(vocab) for vocab in vocabularies.values()),
        'avg_vocab_size': sum(len(vocab) for vocab in vocabularies.values()) / len(vocabularies),
        'slot_stats': {}
    }
    
    # Categorize slots by vocabulary size
    small_vocab = []  # ≤ 20 values
    medium_vocab = []  # 21-50 values  
    large_vocab = []  # > 50 values
    
    for slot_name, vocab in vocabularies.items():
        vocab_size = len(vocab)
        stats['slot_stats'][slot_name] = {
            'vocab_size': vocab_size,
            'sample_values': vocab[:5]
        }
        
        if vocab_size <= 20:
            small_vocab.append(slot_name)
        elif vocab_size <= 50:
            medium_vocab.append(slot_name)
        else:
            large_vocab.append(slot_name)
    
    stats['size_distribution'] = {
        'small_vocab_slots': len(small_vocab),
        'medium_vocab_slots': len(medium_vocab), 
        'large_vocab_slots': len(large_vocab)
    }
    
    return stats


def create_value_to_id_mapping(vocabularies: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Create mapping from values to IDs for each slot"""
    value_to_id = {}
    
    for slot_name, vocab in vocabularies.items():
        value_to_id[slot_name] = {value: idx for idx, value in enumerate(vocab)}
    
    return value_to_id


def main():
    """Build and save slot vocabularies from training data"""
    
    # Paths
    train_data_path = "data/processed/train_instances.json"
    vocab_output_path = "data/processed/slot_vocabularies.json"
    
    if not os.path.exists(train_data_path):
        print(f"❌ Training data not found: {train_data_path}")
        return
    
    # Build vocabularies
    vocabularies = build_slot_vocabularies(train_data_path, min_freq=1)
    
    # Save vocabularies
    save_slot_vocabularies(vocabularies, vocab_output_path)
    
    # Show statistics
    stats = get_vocabulary_statistics(vocabularies)
    
    print(f"\n📊 Vocabulary Statistics:")
    print(f"  Total slots: {stats['total_slots']}")
    print(f"  Total unique values: {stats['total_values']}")
    print(f"  Average vocabulary size: {stats['avg_vocab_size']:.1f}")
    print(f"  Small vocab slots (≤20): {stats['size_distribution']['small_vocab_slots']}")
    print(f"  Medium vocab slots (21-50): {stats['size_distribution']['medium_vocab_slots']}")
    print(f"  Large vocab slots (>50): {stats['size_distribution']['large_vocab_slots']}")
    
    # Show some examples
    print(f"\n📝 Sample Vocabularies:")
    for slot_name in sorted(vocabularies.keys())[:5]:
        vocab = vocabularies[slot_name]
        print(f"  {slot_name}: {len(vocab)} values")
        print(f"    {vocab[:8]}{'...' if len(vocab) > 8 else ''}")
    
    # Create value-to-ID mapping
    value_to_id = create_value_to_id_mapping(vocabularies)
    id_mapping_path = "data/processed/slot_value_to_id.json"
    
    with open(id_mapping_path, 'w') as f:
        json.dump(value_to_id, f, indent=2)
    
    print(f"\n✅ Created vocabulary files:")
    print(f"  - Vocabularies: {vocab_output_path}")
    print(f"  - Value-to-ID mapping: {id_mapping_path}")


if __name__ == "__main__":
    main()