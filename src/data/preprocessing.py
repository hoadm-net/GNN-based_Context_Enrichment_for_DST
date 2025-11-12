#!/usr/bin/env python3
"""
Data preprocessing utilities cho MultiWOZ dataset
Dựa trên các best practices từ DST-STAR, DST-ASSIST, DST-MetaASSIST
"""

import re
import json
import string
from typing import Dict, List, Tuple, Any
from collections import OrderedDict

# Global constants
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']

# Text normalization patterns
TIMEPAT = re.compile(r'\d{1,2}[:]\d{1,2}')
PRICEPAT = re.compile(r'£\d{1,3}[.]?\d{0,2}')

# Replacement mappings for text normalization
REPLACEMENTS = [
    (' phone ', ' telephone '),
    (' address ', ' location '),
    (' postcode ', ' postal code '),
    (' centre ', ' center '),
    (' theatre ', ' theater '),
    (' you\'re ', ' you are '),
    (' you\'ve ', ' you have '),
    (' you\'ll ', ' you will '),
    (' you\'d ', ' you would '),
    (' won\'t ', ' will not '),
    (' can\'t ', ' cannot '),
    (' n\'t ', ' not '),
    (' \'m ', ' am '),
    (' \'re ', ' are '),
    (' \'ve ', ' have '),
    (' \'ll ', ' will '),
    (' \'d ', ' would ')
]

class TextNormalizer:
    """Text normalization utility"""
    
    @staticmethod
    def is_ascii(text: str) -> bool:
        """Check if text contains only ASCII characters"""
        return all(ord(c) < 128 for c in text)
    
    @staticmethod
    def insert_space(token: str, text: str) -> str:
        """Insert spaces around tokens"""
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                    re.match('[0-9]', text[sidx + 1]):
                sidx += 1
                continue
            if text[sidx - 1] != ' ':
                text = text[:sidx] + ' ' + text[sidx:]
                sidx += 1
            if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                text = text[:sidx + 1] + ' ' + text[sidx + 1:]
            sidx += 1
        return text
    
    @staticmethod
    def normalize_text(text: str, clean_value: bool = True) -> str:
        """
        Comprehensive text normalization
        Args:
            text: Input text
            clean_value: Whether to clean values (replace time/price with placeholders)
        """
        # Lower case
        text = text.lower()
        
        # Remove leading/trailing whitespace
        text = re.sub(r'^\s*|\s*$', '', text)
        
        # Hotel domain specific normalization
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
            
            # Normalize postcodes
            ms = re.findall(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
            if ms:
                sidx = 0
                for m in ms:
                    sidx = text.find(m, sidx)
                    eidx = sidx + len(m)
                    text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]
        
        # Fix unicode quotes
        text = re.sub(r"(\u2018|\u2019)", "'", text)
        
        if clean_value:
            # Replace time and price patterns with placeholders
            text = re.sub(TIMEPAT, ' [value_time] ', text)
            text = re.sub(PRICEPAT, ' [value_price] ', text)
        
        # Replace special characters
        text = text.replace(';', ',')
        text = re.sub(r'$\/', '', text)
        text = text.replace('/', ' and ')
        text = text.replace('-', ' ')
        text = re.sub(r'[\"\<>@\(\)]', '', text)
        
        # Insert spaces around punctuation
        for token in ['?', '.', ',', '!']:
            text = TextNormalizer.insert_space(token, text)
        
        # Insert space for possessives
        text = TextNormalizer.insert_space('\'s', text)
        
        # Handle contractions
        text = re.sub(r'^\'', '', text)
        text = re.sub(r'\'$', '', text)
        text = re.sub(r'\'\s', ' ', text)
        text = re.sub(r'\s\'', ' ', text)
        
        # Apply replacement rules
        for from_text, to_text in REPLACEMENTS:
            text = (' ' + text + ' ').replace(from_text, to_text)[1:-1]
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Concatenate numbers
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(r'^\d+$', tokens[i]) and re.match(r'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)
        
        return text.strip()

class BeliefStateProcessor:
    """Process belief states và tạo summaries"""
    
    @staticmethod
    def get_summary_bstate(bstate: Dict, get_domain: bool = False) -> Tuple:
        """
        Tạo summary của belief state thành vector và value list
        Args:
            bstate: Belief state metadata
            get_domain: Nếu True, return active domains
        Returns:
            summary_bstate: Vector encoding của belief state  
            summary_bvalue: List các [slot, value] pairs
        """
        domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
        summary_bstate = []
        summary_bvalue = []
        active_domain = []
        
        for domain in domains:
            domain_active = False
            
            # Process booking slots
            booking = []
            for slot in sorted(bstate[domain]['book'].keys()):
                if slot == 'booked':
                    if len(bstate[domain]['book']['booked']) != 0:
                        booking.append(1)
                    else:
                        booking.append(0)
                else:
                    if bstate[domain]['book'][slot] != "":
                        booking.append(1) 
                        bstate[domain]['book'][slot] = re.sub('[<>]', '|', bstate[domain]['book'][slot])
                        summary_bvalue.append([f"{domain}-book {slot.strip().lower()}", 
                                             TextNormalizer.normalize_text(bstate[domain]['book'][slot].strip().lower(), False)])
                    else:
                        booking.append(0)
            
            # Special handling for train domain
            if domain == 'train':
                if 'people' not in bstate[domain]['book'].keys():
                    booking.append(0)
                if 'ticket' not in bstate[domain]['book'].keys():
                    booking.append(0)
            
            summary_bstate += booking
            
            # Process semi slots
            for slot in bstate[domain]['semi']:
                slot_enc = [0, 0, 0]  # [not mentioned, dontcare, filled]
                if bstate[domain]['semi'][slot] == 'not mentioned':
                    slot_enc[0] = 1
                elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                    slot_enc[1] = 1
                    summary_bvalue.append([f"{domain}-{slot.strip().lower()}", "dontcare"])
                elif bstate[domain]['semi'][slot]:
                    slot_enc[2] = 1
                    bstate[domain]['semi'][slot] = re.sub('[<>]', '|', bstate[domain]['semi'][slot])
                    summary_bvalue.append([f"{domain}-{slot.strip().lower()}", 
                                         TextNormalizer.normalize_text(bstate[domain]['semi'][slot].strip().lower(), False)])
                
                if slot_enc != [0, 0, 0]:
                    domain_active = True
                summary_bstate += slot_enc
            
            # Track active domains
            if domain_active:
                summary_bstate += [1]
                active_domain.append(domain)
            else:
                summary_bstate += [0]
        
        assert len(summary_bstate) == 94  # Fixed size for MultiWOZ
        
        if get_domain:
            return active_domain
        else:
            return summary_bstate, summary_bvalue

class DialogueProcessor:
    """Process individual dialogues"""
    
    def __init__(self, max_length: int = 50):
        self.max_length = max_length
        self.text_normalizer = TextNormalizer()
        self.bs_processor = BeliefStateProcessor()
    
    def analyze_dialogue(self, dialogue: Dict) -> Dict:
        """
        Clean và analyze dialogue
        Args:
            dialogue: Raw dialogue data
        Returns:
            Processed dialogue or None if invalid
        """
        d = dialogue
        
        # Check for odd number of turns
        if len(d['log']) % 2 != 0:
            return None
        
        d_pp = {}
        d_pp['goal'] = d['goal']
        usr_turns = []
        sys_turns = []
        
        for i in range(len(d['log'])):
            # Check turn length
            if len(d['log'][i]['text'].split()) > self.max_length:
                return None
            
            if i % 2 == 0:  # User turn
                text = d['log'][i]['text']
                if not TextNormalizer.is_ascii(text):
                    return None
                usr_turns.append(d['log'][i])
            else:  # System turn
                text = d['log'][i]['text']
                if not TextNormalizer.is_ascii(text):
                    return None
                    
                # Process belief state
                belief_summary, belief_value_summary = self.bs_processor.get_summary_bstate(d['log'][i]['metadata'])
                d['log'][i]['belief_summary'] = str(belief_summary)
                d['log'][i]['belief_value_summary'] = belief_value_summary
                sys_turns.append(d['log'][i])
        
        d_pp['usr_log'] = usr_turns
        d_pp['sys_log'] = sys_turns
        
        return d_pp
    
    def get_dial(self, dialogue: Dict) -> List[Dict]:
        """
        Extract dialogue turns với processing
        Args:
            dialogue: Raw dialogue
        Returns:
            List of processed turns
        """
        dial = []
        d_orig = self.analyze_dialogue(dialogue)
        
        if d_orig is None:
            return None
        
        usr = [t['text'] for t in d_orig['usr_log']]
        sys = [t['text'] for t in d_orig['sys_log']]
        sys_a = [t.get('dialogue_acts', []) for t in d_orig['sys_log']]
        bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
        domain = [t.get('domain', '') for t in d_orig['usr_log']]
        
        for item in zip(usr, sys, sys_a, domain, bvs):
            dial.append({
                'usr': item[0],
                'sys': item[1], 
                'sys_a': item[2],
                'domain': item[3],
                'bvs': item[4]
            })
        
        return dial