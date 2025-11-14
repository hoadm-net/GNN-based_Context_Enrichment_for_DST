"""
Preprocessing MultiWOZ data for Multi-Level GNN-based DST
Creates structured dialogue_history as List[Dict] to enable graph construction
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_ontology(ontology_path: str) -> Dict[str, List[str]]:
    """Load ontology to get all possible slot-value pairs"""
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    return ontology


def extract_belief_state(metadata: Dict) -> Dict[str, str]:
    """
    Extract flat belief state from MultiWOZ metadata format
    Returns: {'domain-slot': 'value', ...}
    """
    belief_state = {}
    
    # MultiWOZ domains
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    
    for domain in domains:
        if domain not in metadata:
            continue
            
        domain_meta = metadata[domain]
        
        # Semi (informable slots - what user wants)
        if 'semi' in domain_meta:
            for slot, value in domain_meta['semi'].items():
                if value and value not in ['', 'not mentioned', 'none']:
                    slot_key = f"{domain}-{slot}"
                    belief_state[slot_key] = value
        
        # Book (requestable slots - booking info)
        if 'book' in domain_meta:
            for slot, value in domain_meta['book'].items():
                if slot == 'booked':  # Skip booked list
                    continue
                if value and value not in ['', 'not mentioned', 'none']:
                    slot_key = f"{domain}-book {slot}"
                    belief_state[slot_key] = value
    
    return belief_state


def get_active_domains(metadata: Dict) -> List[str]:
    """Get list of domains that have non-empty slot values"""
    domains = []
    
    # MultiWOZ domains
    all_domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    
    for domain in all_domains:
        if domain not in metadata:
            continue
            
        domain_meta = metadata[domain]
        
        if 'semi' in domain_meta:
            # Check if any semi slot has value
            for slot, value in domain_meta['semi'].items():
                if value and value not in ['', 'not mentioned', 'none']:
                    if domain not in domains:
                        domains.append(domain)
                    break
        
        if 'book' in domain_meta:
            # Check if any book slot has value
            for slot, value in domain_meta['book'].items():
                if slot == 'booked':
                    continue
                if value and value not in ['', 'not mentioned', 'none']:
                    if domain not in domains:
                        domains.append(domain)
                    break
    
    return domains


def build_turn_dict(turn_idx: int, 
                    user_text: str, 
                    system_text: str,
                    belief_state: Dict[str, str],
                    domains: List[str]) -> Dict[str, Any]:
    """
    Build structured turn dictionary for graph construction
    
    Returns:
        {
            'turn_id': int,
            'user': str,
            'system': str,
            'belief_state': Dict[str, str],  # {'domain-slot': 'value'}
            'domains': List[str],            # Active domains in this turn
            'timestamp': int                 # Turn index for temporal edges
        }
    """
    return {
        'turn_id': turn_idx,
        'user': user_text,
        'system': system_text,
        'belief_state': belief_state,
        'domains': domains,
        'timestamp': turn_idx
    }


def process_dialogue(dialogue_id: str, 
                     dialogue: Dict,
                     ontology: Dict) -> List[Dict[str, Any]]:
    """
    Process a single dialogue into training instances
    Each instance represents prediction task at one turn
    
    Returns list of instances, each with:
        - dialogue_id: str
        - turn_id: int (current turn being predicted)
        - user_utterance: str (current user utterance)
        - system_response: str (current system response)
        - dialogue_history: List[Dict] (all previous turns with structure)
        - previous_belief_state: Dict (belief state from previous turn)
        - current_belief_state: Dict (target belief state for current turn)
        - domains: List[str] (active domains in current turn)
        - is_last_turn: bool
    """
    instances = []
    log = dialogue['log']
    
    # Process turns in pairs (user, system)
    dialogue_history = []
    previous_belief = {}
    
    for turn_idx in range(0, len(log), 2):
        # Check if we have both user and system turns
        if turn_idx + 1 >= len(log):
            break
            
        user_turn = log[turn_idx]
        system_turn = log[turn_idx + 1]
        
        user_text = user_turn['text']
        system_text = system_turn['text']
        
        # Extract current belief state from system turn metadata
        current_belief = extract_belief_state(system_turn['metadata'])
        
        # Get active domains
        active_domains = get_active_domains(system_turn['metadata'])
        
        # Compute delta belief state
        delta_belief = {}
        
        # 1. Add new/updated slots
        for slot, value in current_belief.items():
            if slot not in previous_belief or previous_belief[slot] != value:
                delta_belief[slot] = value
        
        # 2. Mark deleted slots with "none"
        for slot in previous_belief:
            if slot not in current_belief:
                delta_belief[slot] = "none"  # DELETE signal
        
        # Create instance for this turn
        instance = {
            'dialogue_id': dialogue_id,
            'turn_id': turn_idx // 2,  # Actual turn number (0, 1, 2, ...)
            'user_utterance': user_text,
            'system_response': system_text,
            'dialogue_history': dialogue_history.copy(),  # List[Dict] - previous turns
            'previous_belief_state': previous_belief.copy(),
            'current_belief_state': current_belief.copy(),
            'delta_belief_state': delta_belief,  # ADD/UPDATE/DELETE operations
            'domains': active_domains,
            'is_last_turn': (turn_idx + 2 >= len(log))
        }
        instances.append(instance)
        
        # Update history with current turn for next iteration
        turn_dict = build_turn_dict(
            turn_idx=turn_idx // 2,
            user_text=user_text,
            system_text=system_text,
            belief_state=current_belief,
            domains=active_domains
        )
        dialogue_history.append(turn_dict)
        
        # Update previous belief for next turn
        previous_belief = current_belief.copy()
    
    return instances


def create_slot_meta(ontology: Dict) -> List[str]:
    """Create slot metadata list in 'domain-slot' format"""
    # Ontology is already in 'domain-slot': [values] format
    slot_meta = list(ontology.keys())
    return sorted(slot_meta)


def process_dataset(data_path: str, 
                   ontology_path: str,
                   split_files: Dict[str, str],
                   output_dir: str):
    """
    Process MultiWOZ dataset into graph-friendly format
    
    Args:
        data_path: Path to data.json
        ontology_path: Path to ontology.json
        split_files: Dict mapping 'dev'/'test' to list files (train is inferred)
        output_dir: Output directory for processed data
    """
    print("Loading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print("Loading ontology...")
    ontology = load_ontology(ontology_path)
    
    # Load dev/test splits
    dev_ids = set()
    test_ids = set()
    
    if 'dev' in split_files:
        with open(split_files['dev'], 'r') as f:
            dev_ids = set(line.strip() for line in f.readlines())
    
    if 'test' in split_files:
        with open(split_files['test'], 'r') as f:
            test_ids = set(line.strip() for line in f.readlines())
    
    # Train is everything else
    all_ids = set(data.keys())
    train_ids = all_ids - dev_ids - test_ids
    
    splits = {
        'train': list(train_ids),
        'dev': list(dev_ids),
        'test': list(test_ids)
    }
    
    print(f"Split sizes: train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}")
    
    # Process each split
    statistics = {}
    slot_vocab = set()
    
    for split_name, dialogue_ids in splits.items():
        print(f"\nProcessing {split_name} split...")
        instances = []
        
        for idx, dialogue_id in enumerate(dialogue_ids):
            if dialogue_id not in data:
                print(f"Warning: {dialogue_id} not found in data")
                continue
                
            dialogue = data[dialogue_id]
            dialogue_instances = process_dialogue(dialogue_id, dialogue, ontology)
            instances.extend(dialogue_instances)

            # Collect slot names from generated instances for comprehensive slot meta
            for inst in dialogue_instances:
                for state in (
                    inst.get('previous_belief_state', {}),
                    inst.get('current_belief_state', {}),
                    inst.get('delta_belief_state', {})
                ):
                    if isinstance(state, dict):
                        slot_vocab.update(state.keys())
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dialogue_ids)} dialogues, {len(instances)} instances")
        
        # Save instances
        output_file = os.path.join(output_dir, f"{split_name}_instances.json")
        with open(output_file, 'w') as f:
            json.dump(instances, f, indent=2)
        
        print(f"✓ Saved {len(instances)} instances to {output_file}")
        
        # Collect statistics
        num_turns_with_history = sum(1 for inst in instances if len(inst['dialogue_history']) > 0)
        avg_history_length = sum(len(inst['dialogue_history']) for inst in instances) / len(instances) if instances else 0
        
        statistics[split_name] = {
            'num_dialogues': len(dialogue_ids),
            'num_instances': len(instances),
            'num_turns_with_history': num_turns_with_history,
            'avg_history_length': avg_history_length,
            'avg_turns_per_dialogue': len(instances) / len(dialogue_ids) if dialogue_ids else 0
        }
    
    # Save slot meta (include all observed slots; fallback to ontology if empty)
    if not slot_vocab:
        slot_vocab = set(create_slot_meta(ontology))
    slot_meta = sorted(slot_vocab)
    slot_meta_file = os.path.join(output_dir, 'slot_meta.json')
    with open(slot_meta_file, 'w') as f:
        json.dump(slot_meta, f, indent=2)
    print(f"\n✓ Saved {len(slot_meta)} slots to {slot_meta_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved statistics to {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Slot vocabulary size: {len(slot_meta)}")
    for split_name, stats in statistics.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Dialogues: {stats['num_dialogues']}")
        print(f"  Instances: {stats['num_instances']}")
        print(f"  Avg turns/dialogue: {stats['avg_turns_per_dialogue']:.2f}")
        print(f"  Avg history length: {stats['avg_history_length']:.2f}")
        print(f"  Turns with history: {stats['num_turns_with_history']}")


def verify_format(instances_file: str):
    """Verify that processed data has correct format for graph construction"""
    print(f"\nVerifying format of {instances_file}...")
    
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    
    if not instances:
        print("  ✗ No instances found!")
        return False
    
    inst = instances[0]
    
    # Check keys
    required_keys = ['dialogue_id', 'turn_id', 'user_utterance', 'system_response',
                    'dialogue_history', 'previous_belief_state', 'current_belief_state',
                    'delta_belief_state', 'domains', 'is_last_turn']
    
    for key in required_keys:
        if key not in inst:
            print(f"  ✗ Missing key: {key}")
            return False
    
    # Check dialogue_history is List[Dict]
    if not isinstance(inst['dialogue_history'], list):
        print(f"  ✗ dialogue_history is not a list! Type: {type(inst['dialogue_history'])}")
        return False
    
    # Check structure of dialogue_history items (if any)
    if len(inst['dialogue_history']) > 0:
        turn = inst['dialogue_history'][0]
        if not isinstance(turn, dict):
            print(f"  ✗ dialogue_history items are not dicts! Type: {type(turn)}")
            return False
        
        turn_required_keys = ['turn_id', 'user', 'system', 'belief_state', 'domains', 'timestamp']
        for key in turn_required_keys:
            if key not in turn:
                print(f"  ✗ Turn dict missing key: {key}")
                return False
    
    # Check belief states are dicts
    if not isinstance(inst['previous_belief_state'], dict):
        print(f"  ✗ previous_belief_state is not dict!")
        return False
    
    if not isinstance(inst['current_belief_state'], dict):
        print(f"  ✗ current_belief_state is not dict!")
        return False
    
    print("  ✓ Format verification passed!")
    print(f"  ✓ Sample instance keys: {list(inst.keys())}")
    print(f"  ✓ dialogue_history type: List[Dict] with {len(inst['dialogue_history'])} turns")
    
    if len(inst['dialogue_history']) > 0:
        print(f"  ✓ Turn dict keys: {list(inst['dialogue_history'][0].keys())}")
    
    return True


if __name__ == "__main__":
    # Paths
    raw_dir = "data/raw"
    output_dir = "data/processed_graph"
    
    data_path = os.path.join(raw_dir, "data.json")
    ontology_path = os.path.join(raw_dir, "ontology.json")
    
    split_files = {
        'dev': os.path.join(raw_dir, "valListFile.json"),
        'test': os.path.join(raw_dir, "testListFile.json")
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset
    process_dataset(
        data_path=data_path,
        ontology_path=ontology_path,
        split_files=split_files,
        output_dir=output_dir
    )
    
    # Verify format
    print("\n" + "="*60)
    print("FORMAT VERIFICATION")
    print("="*60)
    verify_format(os.path.join(output_dir, "train_instances.json"))
    verify_format(os.path.join(output_dir, "dev_instances.json"))
    verify_format(os.path.join(output_dir, "test_instances.json"))
    
    print("\n✓ Preprocessing complete! Data ready for graph construction.")
