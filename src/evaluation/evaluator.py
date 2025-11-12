#!/usr/bin/env python3
"""
Evaluation framework for Dialogue State Tracking
Implements standard DST evaluation metrics: Joint Goal Accuracy, Slot Accuracy, Turn Accuracy
Based on evaluation methods from DST-STAR, DST-ASSIST, and DST-MetaASSIST
"""

import json
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from copy import deepcopy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DSTEvaluator:
    """Dialogue State Tracking Evaluator"""
    
    def __init__(self, slot_meta: List[str]):
        """
        Initialize evaluator with slot metadata
        
        Args:
            slot_meta: List of slot names (e.g., ['hotel-name', 'restaurant-food'])
        """
        self.slot_meta = slot_meta
        self.num_slots = len(slot_meta)
        
        # Extract domains from slot meta
        self.domains = list(set([slot.split('-')[0] for slot in slot_meta]))
        self.domain_slots = defaultdict(list)
        for slot in slot_meta:
            domain = slot.split('-')[0]
            self.domain_slots[domain].append(slot)
            
        logger.info(f"Initialized evaluator with {self.num_slots} slots across {len(self.domains)} domains")
    
    def normalize_state_value(self, value: str) -> str:
        """Normalize state value for comparison"""
        if not value or value.lower() in ['none', 'not mentioned', '']:
            return 'none'
        if value.lower() in ['dont care', 'dontcare', "don't care", "do not care"]:
            return 'dontcare'
        return value.lower().strip()
    
    def state_to_dict(self, belief_state: List[List[str]]) -> Dict[str, str]:
        """Convert belief state list to dictionary format"""
        state_dict = {}
        
        # Initialize with 'none' for all slots
        for slot in self.slot_meta:
            state_dict[slot] = 'none'
        
        # Fill in actual values
        for slot_value_pair in belief_state:
            if len(slot_value_pair) == 2:
                slot, value = slot_value_pair
                if slot in self.slot_meta:
                    state_dict[slot] = self.normalize_state_value(value)
                    
        return state_dict
    
    def compare_states(self, pred_state: Dict[str, str], 
                      gold_state: Dict[str, str]) -> Tuple[bool, List[bool]]:
        """
        Compare predicted and gold states
        
        Returns:
            joint_match: True if all slots match
            slot_matches: List of boolean matches for each slot
        """
        slot_matches = []
        
        for slot in self.slot_meta:
            pred_val = self.normalize_state_value(pred_state.get(slot, 'none'))
            gold_val = self.normalize_state_value(gold_state.get(slot, 'none'))
            slot_matches.append(pred_val == gold_val)
        
        joint_match = all(slot_matches)
        return joint_match, slot_matches
    
    def evaluate_predictions(self, predictions: List[Dict], 
                           ground_truth: List[Dict],
                           compute_turn_accuracy: bool = True) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries  
            compute_turn_accuracy: Whether to compute turn-level accuracy
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) != Ground truth count ({len(ground_truth)})")
        
        # Initialize counters
        joint_acc_count = 0
        slot_acc_counts = np.zeros(self.num_slots)
        turn_acc_count = 0
        final_joint_acc_count = 0
        final_count = 0
        
        # Per-domain metrics
        domain_joint_acc = {domain: 0 for domain in self.domains}
        domain_counts = {domain: 0 for domain in self.domains}
        
        # Detailed results for analysis
        detailed_results = []
        
        # Track dialogue states for turn accuracy
        dialogue_states = {}
        
        wall_times = []
        
        for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
            start_time = time.perf_counter()
            
            # Extract required fields
            dialogue_id = pred.get('dialogue_id', f'dialogue_{i}')
            turn_id = pred.get('turn_id', 0)
            is_last_turn = pred.get('is_last_turn', False)
            
            # Get predicted and gold belief states
            pred_belief = pred.get('belief_state', [])
            gold_belief = gold.get('belief_state', [])
            
            # Convert to dictionary format
            pred_state_dict = self.state_to_dict(pred_belief)
            gold_state_dict = self.state_to_dict(gold_belief)
            
            # Compare states
            joint_match, slot_matches = self.compare_states(pred_state_dict, gold_state_dict)
            
            # Update counters
            if joint_match:
                joint_acc_count += 1
                
            slot_acc_counts += np.array(slot_matches, dtype=int)
            
            # Per-domain accuracy
            for domain in self.domains:
                domain_slots = self.domain_slots[domain]
                if any(gold_state_dict.get(slot, 'none') != 'none' for slot in domain_slots):
                    domain_counts[domain] += 1
                    domain_slots_match = all(
                        pred_state_dict.get(slot, 'none') == gold_state_dict.get(slot, 'none')
                        for slot in domain_slots
                    )
                    if domain_slots_match:
                        domain_joint_acc[domain] += 1
            
            # Turn-level accuracy (only changed slots)
            if compute_turn_accuracy:
                # Get previous state for this dialogue
                prev_state = dialogue_states.get(dialogue_id, {slot: 'none' for slot in self.slot_meta})
                
                # Find slots that changed in this turn
                gold_changes = []
                pred_changes = []
                
                for slot in self.slot_meta:
                    if gold_state_dict[slot] != prev_state[slot]:
                        gold_changes.append(f"{slot}-{gold_state_dict[slot]}")
                    if pred_state_dict[slot] != prev_state[slot]:
                        pred_changes.append(f"{slot}-{pred_state_dict[slot]}")
                
                # Check if turn changes match
                if set(gold_changes) == set(pred_changes):
                    turn_acc_count += 1
                
                # Update dialogue state
                dialogue_states[dialogue_id] = gold_state_dict.copy()
            
            # Final turn accuracy
            if is_last_turn:
                final_count += 1
                if joint_match:
                    final_joint_acc_count += 1
            
            # Store detailed result
            result_detail = {
                'dialogue_id': dialogue_id,
                'turn_id': turn_id,
                'joint_match': joint_match,
                'slot_matches': slot_matches,
                'predicted_state': pred_state_dict,
                'gold_state': gold_state_dict,
                'is_last_turn': is_last_turn
            }
            detailed_results.append(result_detail)
            
            end_time = time.perf_counter()
            wall_times.append(end_time - start_time)
        
        # Calculate final metrics
        num_examples = len(predictions)
        
        metrics = {
            # Core metrics
            'joint_accuracy': joint_acc_count / num_examples,
            'slot_accuracy': slot_acc_counts / num_examples,
            'average_slot_accuracy': np.mean(slot_acc_counts / num_examples),
            
            # Latency
            'avg_prediction_time_ms': np.mean(wall_times) * 1000,
            'total_examples': num_examples,
        }
        
        # Turn accuracy
        if compute_turn_accuracy:
            metrics['turn_accuracy'] = turn_acc_count / num_examples
        
        # Final turn accuracy
        if final_count > 0:
            metrics['final_joint_accuracy'] = final_joint_acc_count / final_count
            metrics['final_turn_count'] = final_count
        
        # Per-domain accuracy
        domain_accuracies = {}
        for domain in self.domains:
            if domain_counts[domain] > 0:
                domain_accuracies[domain] = domain_joint_acc[domain] / domain_counts[domain]
            else:
                domain_accuracies[domain] = 0.0
        metrics['domain_accuracy'] = domain_accuracies
        
        # Per-slot accuracy details
        slot_accuracies = {}
        for i, slot in enumerate(self.slot_meta):
            slot_accuracies[slot] = float(slot_acc_counts[i] / num_examples)
        metrics['slot_accuracy_detail'] = slot_accuracies
        
        # Store detailed results
        metrics['detailed_results'] = detailed_results
        
        return metrics
    
    def print_evaluation_results(self, metrics: Dict[str, Any], 
                               model_name: str = "Model", 
                               epoch: int = None):
        """Print formatted evaluation results"""
        
        print("=" * 60)
        if epoch is not None:
            print(f"Evaluation Results - {model_name} (Epoch {epoch})")
        else:
            print(f"Evaluation Results - {model_name}")
        print("=" * 60)
        
        # Core metrics
        print(f"Joint Goal Accuracy: {metrics['joint_accuracy']:.4f}")
        print(f"Average Slot Accuracy: {metrics['average_slot_accuracy']:.4f}")
        
        if 'turn_accuracy' in metrics:
            print(f"Turn Accuracy: {metrics['turn_accuracy']:.4f}")
            
        if 'final_joint_accuracy' in metrics:
            print(f"Final Joint Accuracy: {metrics['final_joint_accuracy']:.4f}")
            print(f"Final Turn Count: {metrics['final_turn_count']}")
        
        # Performance
        print(f"Average Prediction Time: {metrics['avg_prediction_time_ms']:.2f} ms")
        print(f"Total Examples: {metrics['total_examples']}")
        
        # Per-domain accuracy
        print("\nPer-Domain Joint Accuracy:")
        for domain, acc in metrics['domain_accuracy'].items():
            print(f"  {domain}: {acc:.4f}")
        
        # Top-5 worst performing slots
        slot_accs = [(slot, acc) for slot, acc in metrics['slot_accuracy_detail'].items()]
        slot_accs.sort(key=lambda x: x[1])
        
        print("\nTop-5 Most Challenging Slots:")
        for slot, acc in slot_accs[:5]:
            print(f"  {slot}: {acc:.4f}")
            
        print("=" * 60)
    
    def save_evaluation_results(self, metrics: Dict[str, Any], 
                              output_path: str,
                              include_detailed: bool = False):
        """Save evaluation results to file"""
        
        # Create a clean version without detailed results for main file
        clean_metrics = {k: v for k, v in metrics.items() if k != 'detailed_results'}
        
        # Convert numpy arrays to lists for JSON serialization
        if 'slot_accuracy' in clean_metrics:
            clean_metrics['slot_accuracy'] = clean_metrics['slot_accuracy'].tolist()
        
        with open(output_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        
        # Save detailed results separately if requested
        if include_detailed and 'detailed_results' in metrics:
            detailed_path = output_path.replace('.json', '_detailed.json')
            with open(detailed_path, 'w') as f:
                json.dump(metrics['detailed_results'], f, indent=2)
            logger.info(f"Detailed results saved to {detailed_path}")
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results from multiple models"""
        
        comparison = {
            'models': list(results.keys()),
            'metrics_comparison': {},
            'best_model': {}
        }
        
        # Key metrics to compare
        key_metrics = ['joint_accuracy', 'average_slot_accuracy', 'turn_accuracy']
        
        for metric in key_metrics:
            if all(metric in results[model] for model in results):
                comparison['metrics_comparison'][metric] = {
                    model: results[model][metric] for model in results
                }
                
                # Find best model for this metric
                best_model = max(results.keys(), key=lambda m: results[m][metric])
                comparison['best_model'][metric] = {
                    'model': best_model,
                    'score': results[best_model][metric]
                }
        
        return comparison


def load_evaluation_data(predictions_path: str, ground_truth_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load prediction and ground truth data for evaluation"""
    
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    return predictions, ground_truth


def main():
    """Example usage of the evaluation framework"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DST model predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions file")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth file")
    parser.add_argument("--slot_meta", type=str, required=True,
                        help="Path to slot metadata file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, default="Model",
                        help="Name of the model being evaluated")
    parser.add_argument("--detailed", action='store_true',
                        help="Save detailed results")
    
    args = parser.parse_args()
    
    # Load slot metadata
    with open(args.slot_meta, 'r') as f:
        slot_data = json.load(f)
        slot_meta = slot_data['slot_meta']
    
    # Load evaluation data
    predictions, ground_truth = load_evaluation_data(args.predictions, args.ground_truth)
    
    # Initialize evaluator
    evaluator = DSTEvaluator(slot_meta)
    
    # Run evaluation
    metrics = evaluator.evaluate_predictions(predictions, ground_truth)
    
    # Print results
    evaluator.print_evaluation_results(metrics, args.model_name)
    
    # Save results
    evaluator.save_evaluation_results(metrics, args.output, include_detailed=args.detailed)


if __name__ == "__main__":
    main()