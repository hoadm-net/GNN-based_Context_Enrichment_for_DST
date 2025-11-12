#!/usr/bin/env python3
"""
Data preprocessing pipeline for MultiWOZ 2.4 dataset
Based on preprocessing from DST-STAR, DST-ASSIST, and DST-MetaASSIST
"""

import json
import os
import re
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']
MAX_LENGTH = 50  # max turn length in words

# Text normalization patterns
TIMEPAT = re.compile(r'\d{1,2}[:]\d{1,2}')
PRICEPAT = re.compile(r'[£]\d{1,3}[\.]\d{1,2}')

# Slot name normalization mapping
SLOT_NORMALIZE_MAP = {
    "pricerange": "price range",
    "leaveat": "leave at", 
    "arriveby": "arrive by"
}

class MultiWOZPreprocessor:
    """Main preprocessor class for MultiWOZ dataset"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.slot_meta = []
        self.ontology = {}
        self.label_maps = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def normalize_text(self, text: str, clean_value: bool = True) -> str:
        """Normalize text following MultiWOZ preprocessing"""
        # Lower case
        text = text.lower()
        
        # Remove leading/trailing whitespace
        text = re.sub(r'^\s*|\s*$', '', text)
        
        # Domain-specific normalizations
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)
        text = re.sub(r"guesthouse", "guest house", text)
        
        if clean_value:
            # Normalize phone numbers
            ms = re.findall(r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
            if ms:
                sidx = 0
                for m in ms:
                    sidx = text.find(m[0], sidx)
                    if sidx > 0 and text[sidx - 1] == '(':
                        sidx -= 1
                    eidx = text.find(m[-1], sidx) + len(m[-1])
                    text = text[:sidx] + ''.join(m) + text[eidx:]
            
            # Normalize postcode
            ms = re.findall(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
            if ms:
                sidx = 0
                for m in ms:
                    sidx = text.find(m, sidx)
                    eidx = sidx + len(m)
                    text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]
        
        # Unicode normalization
        text = re.sub(u"(\u2018|\u2019)", "'", text)
        
        if clean_value:
            # Replace time and price patterns
            text = re.sub(TIMEPAT, ' [value_time] ', text)
            text = re.sub(PRICEPAT, ' [value_price] ', text)
        
        # Replace special characters
        text = text.replace(';', ',')
        text = text.replace('/', ' and ')
        text = text.replace('-', ' ')
        text = re.sub(r'[\"\<>@\(\)]', '', text)
        
        # Insert spaces around punctuation
        for token in ['?', '.', ',', '!']:
            text = self._insert_space(token, text)
        
        # Handle contractions
        text = self._insert_space('\'s', text)
        text = re.sub(r'^\'', '', text)
        text = re.sub(r'\'$', '', text)
        text = re.sub(r'\'\s', ' ', text)
        text = re.sub(r'\s\'', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _insert_space(self, token: str, text: str) -> str:
        """Insert spaces around tokens"""
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            
            if (sidx + 1 < len(text) and sidx > 0 and 
                re.match(r'[0-9]', text[sidx - 1]) and 
                re.match(r'[0-9]', text[sidx + 1])):
                sidx += 1
                continue
                
            if sidx > 0 and text[sidx - 1] != ' ':
                text = text[:sidx] + ' ' + text[sidx:]
                sidx += 1
                
            if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                text = text[:sidx + len(token)] + ' ' + text[sidx + len(token):]
                
            sidx += 1
            
        return text
    
    def normalize_slot_value(self, slot: str, value: str) -> str:
        """Normalize slot values"""
        if not value or value.lower() in ['none', 'not mentioned', '']:
            return 'none'
            
        if value.lower() in ['dont care', 'dontcare', "don't care", "do not care"]:
            return 'dontcare'
            
        # Normalize the value text
        normalized = self.normalize_text(value, clean_value=False)
        
        # Handle special cases
        if 'time' in slot.lower():
            # Time normalization
            normalized = re.sub(r'(\d{1,2})[:\.](\d{2})', r'\1:\2', normalized)
            
        elif 'price' in slot.lower():
            # Price normalization  
            normalized = re.sub(r'[£$]', '', normalized)
            if normalized in ['cheap', 'moderate', 'expensive']:
                return normalized
                
        return normalized
    
    def extract_slot_meta(self, ontology_path: str) -> List[str]:
        """Extract slot metadata from ontology"""
        with open(ontology_path, 'r') as f:
            ontology = json.load(f)
            
        slot_meta = []
        for slot, values in ontology.items():
            # Only include slots from experiment domains
            domain = slot.split('-')[0]
            if domain in EXPERIMENT_DOMAINS:
                slot_meta.append(slot)
                
        # Sort for consistency
        slot_meta.sort()
        
        logger.info(f"Extracted {len(slot_meta)} slots from ontology")
        return slot_meta
    
    def get_belief_state_summary(self, metadata: Dict) -> Tuple[List[int], List[List[str]]]:
        """Get belief state summary from metadata"""
        domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
        summary_bstate = []
        summary_bvalue = []
        
        for domain in domains:
            if domain not in metadata:
                continue
                
            domain_active = False
            
            # Process booking information
            booking = []
            if 'book' in metadata[domain]:
                for slot in sorted(metadata[domain]['book'].keys()):
                    if slot == 'booked':
                        if len(metadata[domain]['book']['booked']) != 0:
                            booking.append(1)
                        else:
                            booking.append(0)
                    else:
                        if metadata[domain]['book'][slot] != "":
                            booking.append(1)
                            normalized_val = self.normalize_text(
                                metadata[domain]['book'][slot].strip().lower(), False
                            )
                            summary_bvalue.append([f"{domain}-book {slot.strip().lower()}", normalized_val])
                        else:
                            booking.append(0)
                            
            summary_bstate += booking
            
            # Process semi information
            if 'semi' in metadata[domain]:
                for slot in metadata[domain]['semi']:
                    slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
                    
                    if metadata[domain]['semi'][slot] == 'not mentioned':
                        slot_enc[0] = 1
                    elif metadata[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                        slot_enc[1] = 1
                        summary_bvalue.append([f"{domain}-{slot.strip().lower()}", "dontcare"])
                    elif metadata[domain]['semi'][slot]:
                        slot_enc[2] = 1
                        normalized_val = self.normalize_text(
                            metadata[domain]['semi'][slot].strip().lower(), False
                        )
                        summary_bvalue.append([f"{domain}-{slot.strip().lower()}", normalized_val])
                        
                    if slot_enc != [0, 0, 0]:
                        domain_active = True
                        
                    summary_bstate += slot_enc
            
            # Mark domain as active or inactive
            if domain_active:
                summary_bstate += [1]
            else:
                summary_bstate += [0]
                
        return summary_bstate, summary_bvalue
    
    def analyze_dialogue(self, dialogue: Dict, maxlen: int = MAX_LENGTH) -> Optional[Dict]:
        """Analyze and clean dialogue"""
        d = dialogue
        
        # Check for odd number of turns
        if len(d['log']) % 2 != 0:
            logger.warning("Odd number of turns in dialogue")
            return None
            
        d_processed = {}
        d_processed['goal'] = d['goal']
        usr_turns = []
        sys_turns = []
        
        for i in range(len(d['log'])):
            # Check turn length
            if len(d['log'][i]['text'].split()) > maxlen:
                logger.warning(f"Turn too long: {len(d['log'][i]['text'].split())} words")
                return None
                
            if i % 2 == 0:  # User turn
                text = d['log'][i]['text']
                if not self._is_ascii(text):
                    logger.warning("Non-ASCII text found")
                    return None
                usr_turns.append(d['log'][i])
            else:  # System turn
                text = d['log'][i]['text']
                if not self._is_ascii(text):
                    logger.warning("Non-ASCII text found")
                    return None
                    
                # Get belief state summary
                belief_summary, belief_value_summary = self.get_belief_state_summary(d['log'][i]['metadata'])
                d['log'][i]['belief_summary'] = str(belief_summary)
                d['log'][i]['belief_value_summary'] = belief_value_summary
                sys_turns.append(d['log'][i])
        
        d_processed['usr_log'] = usr_turns
        d_processed['sys_log'] = sys_turns
        
        return d_processed
    
    def _is_ascii(self, text: str) -> bool:
        """Check if text contains only ASCII characters"""
        return all(ord(c) < 128 for c in text)
    
    def process_dialogues(self, data_path: str) -> Dict:
        """Process all dialogues in the dataset"""
        logger.info(f"Loading dialogues from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        processed_data = {}
        processed_count = 0
        
        for dialogue_name, dialogue in tqdm(data.items(), desc="Processing dialogues"):
            # Extract domains from goal
            domains = []
            for dom_k, dom_v in dialogue['goal'].items():
                if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:
                    domains.append(dom_k)
            
            # Only process dialogues with experiment domains
            if not any(domain in EXPERIMENT_DOMAINS for domain in domains):
                continue
                
            # Analyze dialogue
            processed_dialogue = self.analyze_dialogue(dialogue)
            if processed_dialogue is not None:
                processed_dialogue['domains'] = domains
                processed_data[dialogue_name] = processed_dialogue
                processed_count += 1
        
        logger.info(f"Successfully processed {processed_count}/{len(data)} dialogues")
        return processed_data
    
    def create_training_instances(self, processed_data: Dict, 
                                val_list: List[str], test_list: List[str]) -> Dict[str, List]:
        """Create training instances for each split"""
        
        instances = {
            'train': [],
            'dev': [],
            'test': []
        }
        
        for dialogue_name, dialogue in tqdm(processed_data.items(), desc="Creating instances"):
            # Determine split
            if dialogue_name in test_list:
                split = 'test'
            elif dialogue_name in val_list:
                split = 'dev'
            else:
                split = 'train'
            
            # Extract turns
            usr_turns = [turn['text'] for turn in dialogue['usr_log']]
            sys_turns = [turn['text'] for turn in dialogue['sys_log']]
            belief_states = [turn['belief_value_summary'] for turn in dialogue['sys_log']]
            
            # Create instances for each turn
            dialogue_history = []
            
            for turn_idx, (usr_turn, sys_turn, belief_state) in enumerate(
                zip(usr_turns, sys_turns, belief_states)
            ):
                # Create turn instance
                instance = {
                    'dialogue_id': dialogue_name,
                    'turn_id': turn_idx,
                    'user_utterance': self.normalize_text(usr_turn),
                    'system_response': self.normalize_text(sys_turn) if turn_idx > 0 else "",
                    'dialogue_history': ' '.join(dialogue_history[-10:]),  # Keep last 10 turns
                    'belief_state': belief_state,
                    'domains': dialogue['domains'],
                    'is_last_turn': turn_idx == len(usr_turns) - 1
                }
                
                instances[split].append(instance)
                
                # Update dialogue history
                if turn_idx > 0:
                    dialogue_history.append(f"System: {sys_turn}")
                dialogue_history.append(f"User: {usr_turn}")
        
        # Log statistics
        for split, split_instances in instances.items():
            logger.info(f"{split}: {len(split_instances)} instances")
            
        return instances
    
    def save_processed_data(self, instances: Dict[str, List], slot_meta: List[str]):
        """Save processed data to files"""
        
        # Save instances
        for split, split_instances in instances.items():
            output_path = os.path.join(self.output_dir, f"{split}_instances.json")
            with open(output_path, 'w') as f:
                json.dump(split_instances, f, indent=2)
            logger.info(f"Saved {len(split_instances)} {split} instances to {output_path}")
        
        # Save slot metadata
        slot_meta_path = os.path.join(self.output_dir, "slot_meta.json")
        with open(slot_meta_path, 'w') as f:
            json.dump({"slot_meta": slot_meta}, f, indent=2)
        logger.info(f"Saved slot metadata to {slot_meta_path}")
        
        # Create statistics
        stats = {
            'total_instances': sum(len(instances[split]) for split in instances),
            'splits': {split: len(instances[split]) for split in instances},
            'num_slots': len(slot_meta),
            'domains': EXPERIMENT_DOMAINS
        }
        
        stats_path = os.path.join(self.output_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting MultiWOZ 2.4 preprocessing...")
        
        # Paths
        data_path = os.path.join(self.data_dir, "data.json")
        ontology_path = os.path.join(self.data_dir, "ontology.json")
        val_list_path = os.path.join(self.data_dir, "valListFile.json")
        test_list_path = os.path.join(self.data_dir, "testListFile.json")
        
        # Load split lists
        with open(val_list_path, 'r') as f:
            val_list = [line.strip() for line in f.readlines()]
        
        with open(test_list_path, 'r') as f:
            test_list = [line.strip() for line in f.readlines()]
        
        logger.info(f"Val set: {len(val_list)} dialogues")
        logger.info(f"Test set: {len(test_list)} dialogues")
        
        # Extract slot metadata
        slot_meta = self.extract_slot_meta(ontology_path)
        
        # Process dialogues
        processed_data = self.process_dialogues(data_path)
        
        # Create training instances
        instances = self.create_training_instances(processed_data, val_list, test_list)
        
        # Save processed data
        self.save_processed_data(instances, slot_meta)
        
        logger.info("Preprocessing completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiWOZ 2.4 dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", 
                        help="Path to raw MultiWOZ data")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Path to save processed data")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MultiWOZPreprocessor(args.data_dir, args.output_dir)
    
    # Run preprocessing
    preprocessor.run_preprocessing()


if __name__ == "__main__":
    main()