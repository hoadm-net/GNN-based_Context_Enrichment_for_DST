#!/usr/bin/env python3
"""
Main preprocessing script cho MultiWOZ dataset
Tạo processed data ready for training/evaluation
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import pickle

from preprocessing import DialogueProcessor, TextNormalizer, EXPERIMENT_DOMAINS, IGNORE_KEYS_IN_GOAL

class MultiWOZPreprocessor:
    """Main preprocessor for MultiWOZ dataset"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dialogue_processor = DialogueProcessor()
        
    def load_raw_data(self) -> Dict:
        """Load raw MultiWOZ data"""
        data_file = self.raw_data_dir / "data.json"
        ontology_file = self.raw_data_dir / "ontology.json"
        
        print("Loading raw data...")
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        with open(ontology_file, 'r') as f:
            ontology = json.load(f)
            
        return data, ontology
    
    def load_splits(self) -> Dict[str, List[str]]:
        """Load train/val/test splits"""
        splits = {}
        
        # Load test split
        test_file = self.raw_data_dir / "testListFile.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                splits['test'] = [line.strip() for line in f]
        
        # Load val split  
        val_file = self.raw_data_dir / "valListFile.json"
        if val_file.exists():
            with open(val_file, 'r') as f:
                splits['val'] = [line.strip() for line in f]
        
        return splits
    
    def create_slot_meta(self, ontology: Dict) -> List[str]:
        """Create slot metadata từ ontology"""
        slot_meta = []
        
        for slot, values in ontology.items():
            if slot == "train-book ticket":
                continue  # Skip invalid slot
            slot_meta.append(slot)
            
        return sorted(slot_meta)
    
    def process_dialogue_for_dst(self, dialogue_name: str, dialogue: Dict, 
                                domains: List[str]) -> Dict:
        """
        Process một dialogue cho DST task
        Args:
            dialogue_name: Dialogue ID
            dialogue: Raw dialogue data
            domains: Active domains trong dialogue
        Returns:
            Processed dialogue structure
        """
        dial = self.dialogue_processor.get_dial(dialogue)
        if dial is None:
            return None
            
        processed_dialogue = {
            'dialogue_idx': dialogue_name,
            'domains': list(set(domains)),
            'dialogue': []
        }
        
        last_bs = []
        
        for turn_i, turn in enumerate(dial):
            turn_dialog = {
                'turn_idx': turn_i,
                'system_transcript': dial[turn_i-1]['sys'] if turn_i > 0 else "",
                'transcript': turn['usr'],
                'belief_state': [{"slots": [s], "act": "inform"} for s in turn['bvs']],
                'system_acts': turn['sys_a'],
                'domain': turn['domain']
            }
            
            # Calculate turn label (changes from last turn)
            turn_dialog['turn_label'] = []
            current_bs = turn_dialog['belief_state']
            for bs in current_bs:
                if bs not in last_bs:
                    if bs["slots"]:  # Check if slots exist
                        turn_dialog['turn_label'].append(bs["slots"][0])
            
            last_bs = current_bs
            processed_dialogue['dialogue'].append(turn_dialog)
            
        return processed_dialogue
    
    def preprocess_dataset(self) -> Dict[str, Any]:
        """Main preprocessing function"""
        print("Starting dataset preprocessing...")
        
        # Load raw data
        data, ontology = self.load_raw_data()
        splits = self.load_splits()
        
        # Create slot metadata
        slot_meta = self.create_slot_meta(ontology)
        
        # Initialize processed data structure
        processed_data = {
            'ontology': ontology,
            'slot_meta': slot_meta,
            'splits': splits,
            'dialogues': {
                'train': [],
                'val': [],
                'test': []
            },
            'statistics': {}
        }
        
        # Determine splits
        test_ids = set(splits.get('test', []))
        val_ids = set(splits.get('val', []))
        train_ids = set(data.keys()) - test_ids - val_ids
        
        print(f"Processing {len(data)} dialogues...")
        
        valid_count = 0
        invalid_count = 0
        
        for dialogue_name, dialogue in tqdm(data.items()):
            # Extract domains from goal
            domains = []
            for dom_k, dom_v in dialogue['goal'].items():
                if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:
                    domains.append(dom_k)
            
            # Process dialogue
            processed_dialogue = self.process_dialogue_for_dst(
                dialogue_name, dialogue, domains
            )
            
            if processed_dialogue is None:
                invalid_count += 1
                continue
                
            valid_count += 1
            
            # Assign to appropriate split
            if dialogue_name in test_ids:
                processed_data['dialogues']['test'].append(processed_dialogue)
            elif dialogue_name in val_ids:
                processed_data['dialogues']['val'].append(processed_dialogue)
            else:
                processed_data['dialogues']['train'].append(processed_dialogue)
        
        # Calculate statistics
        processed_data['statistics'] = {
            'total_dialogues': len(data),
            'valid_dialogues': valid_count,
            'invalid_dialogues': invalid_count,
            'train_dialogues': len(processed_data['dialogues']['train']),
            'val_dialogues': len(processed_data['dialogues']['val']),
            'test_dialogues': len(processed_data['dialogues']['test']),
            'domains': sorted(list(set([d for dialogue in data.values() 
                                      for d in dialogue['goal'].keys()
                                      if d not in IGNORE_KEYS_IN_GOAL]))),
            'num_slots': len(slot_meta)
        }
        
        return processed_data
    
    def save_processed_data(self, processed_data: Dict):
        """Save processed data to files"""
        print("Saving processed data...")
        
        # Save full processed data
        full_file = self.processed_data_dir / "processed_data.pkl"
        with open(full_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Save individual splits as JSON
        for split in ['train', 'val', 'test']:
            split_file = self.processed_data_dir / f"{split}_dials.json"
            with open(split_file, 'w') as f:
                json.dump(processed_data['dialogues'][split], f, indent=2)
        
        # Save metadata
        metadata = {
            'ontology': processed_data['ontology'],
            'slot_meta': processed_data['slot_meta'],
            'statistics': processed_data['statistics']
        }
        
        metadata_file = self.processed_data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed data saved to {self.processed_data_dir}")
        
    def print_statistics(self, processed_data: Dict):
        """Print processing statistics"""
        stats = processed_data['statistics']
        
        print("\n=== Preprocessing Statistics ===")
        print(f"Total dialogues: {stats['total_dialogues']}")
        print(f"Valid dialogues: {stats['valid_dialogues']}")
        print(f"Invalid dialogues: {stats['invalid_dialogues']}")
        print(f"Success rate: {stats['valid_dialogues']/stats['total_dialogues']*100:.1f}%")
        
        print(f"\nSplit distribution:")
        print(f"- Train: {stats['train_dialogues']} dialogues")
        print(f"- Validation: {stats['val_dialogues']} dialogues")
        print(f"- Test: {stats['test_dialogues']} dialogues")
        
        print(f"\nDomains: {stats['domains']}")
        print(f"Number of slots: {stats['num_slots']}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess MultiWOZ dataset')
    parser.add_argument('--raw_data_dir', default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--processed_data_dir', default='data/processed', 
                       help='Directory to save processed data')
    
    args = parser.parse_args()
    
    preprocessor = MultiWOZPreprocessor(args.raw_data_dir, args.processed_data_dir)
    
    # Run preprocessing
    processed_data = preprocessor.preprocess_dataset()
    
    # Save results
    preprocessor.save_processed_data(processed_data)
    
    # Print statistics
    preprocessor.print_statistics(processed_data)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()