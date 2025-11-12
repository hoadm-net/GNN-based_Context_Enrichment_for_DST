#!/usr/bin/env python3
"""
Test script to validate the complete data pipeline
Tests data loading, preprocessing, and evaluation framework
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DSTDataLoader
from src.evaluation.evaluator import DSTEvaluator

def test_data_pipeline():
    """Test the complete data pipeline"""
    
    print("=" * 60)
    print("Testing DST Data Pipeline")
    print("=" * 60)
    
    # Check if processed data exists
    data_dir = "data/processed"
    slot_meta_path = "data/processed/slot_meta.json"
    
    if not os.path.exists(slot_meta_path):
        print("❌ Slot metadata not found. Please run preprocessing first.")
        return False
    
    # Test 1: Data Loader
    print("\n1. Testing Data Loader...")
    try:
        data_loader = DSTDataLoader(
            data_dir=data_dir,
            slot_meta_path=slot_meta_path,
            batch_size=4
        )
        
        # Get statistics
        stats = data_loader.get_statistics()
        print(f"✅ Data loader initialized successfully")
        print(f"   Total instances: {stats['total_instances']}")
        print(f"   Number of slots: {stats['num_slots']}")
        
        for split, size in stats['splits'].items():
            print(f"   {split}: {size} instances")
            
        # Test data loading for each split
        for split in ['train', 'dev', 'test']:
            if split in data_loader.datasets and data_loader.datasets[split] is not None:
                dataloader = data_loader.get_dataloader(split)
                
                # Load first batch
                first_batch = next(iter(dataloader))
                print(f"   {split} batch size: {first_batch['batch_size']}")
                print(f"   {split} sample dialogue: {first_batch['dialogue_ids'][0]}")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False
    
    # Test 2: Evaluation Framework
    print("\n2. Testing Evaluation Framework...")
    try:
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_data = json.load(f)
            slot_meta = slot_data['slot_meta']
        
        evaluator = DSTEvaluator(slot_meta)
        print(f"✅ Evaluator initialized with {evaluator.num_slots} slots")
        print(f"   Domains: {evaluator.domains}")
        
        # Create dummy predictions and ground truth for testing
        print("\n   Creating dummy evaluation data...")
        dummy_predictions, dummy_ground_truth = create_dummy_evaluation_data(slot_meta)
        
        # Run evaluation
        metrics = evaluator.evaluate_predictions(dummy_predictions, dummy_ground_truth)
        
        print(f"✅ Evaluation completed successfully")
        print(f"   Joint Accuracy: {metrics['joint_accuracy']:.4f}")
        print(f"   Average Slot Accuracy: {metrics['average_slot_accuracy']:.4f}")
        
        # Test result printing
        evaluator.print_evaluation_results(metrics, model_name="Test Model")
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False
    
    # Test 3: Integration Test
    print("\n3. Testing Integration...")
    try:
        # Load actual dev data for integration test
        dev_path = os.path.join(data_dir, "dev_instances.json")
        if os.path.exists(dev_path):
            with open(dev_path, 'r') as f:
                dev_instances = json.load(f)
            
            # Take first 10 instances for quick test
            test_instances = dev_instances[:10]
            
            # Create predictions (copy ground truth with some noise)
            predictions = []
            for instance in test_instances:
                pred = {
                    'dialogue_id': instance['dialogue_id'],
                    'turn_id': instance['turn_id'],
                    'belief_state': instance['belief_state'].copy(),
                    'is_last_turn': instance.get('is_last_turn', False)
                }
                
                # Add some noise to predictions
                if len(pred['belief_state']) > 0 and np.random.random() < 0.2:
                    # Randomly modify 20% of predictions
                    idx = np.random.randint(0, len(pred['belief_state']))
                    pred['belief_state'][idx][1] = 'corrupted_value'
                
                predictions.append(pred)
            
            # Evaluate
            metrics = evaluator.evaluate_predictions(predictions, test_instances)
            print(f"✅ Integration test completed")
            print(f"   Tested on {len(test_instances)} instances")
            print(f"   Joint Accuracy: {metrics['joint_accuracy']:.4f}")
            
        else:
            print("⚠️  Dev data not found, skipping integration test")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Data pipeline is ready.")
    print("=" * 60)
    
    return True


def create_dummy_evaluation_data(slot_meta, num_samples=50):
    """Create dummy evaluation data for testing"""
    
    predictions = []
    ground_truth = []
    
    # Sample values for testing
    sample_values = ['none', 'dontcare', 'restaurant_value', 'hotel_value', 'attraction_value']
    
    for i in range(num_samples):
        # Create ground truth
        gt_belief_state = []
        pred_belief_state = []
        
        # Randomly assign values to some slots
        active_slots = np.random.choice(slot_meta, size=np.random.randint(1, min(5, len(slot_meta))), replace=False)
        
        for slot in active_slots:
            value = np.random.choice(sample_values)
            gt_belief_state.append([slot, value])
            
            # Create prediction (with some noise)
            if np.random.random() < 0.8:  # 80% accuracy
                pred_belief_state.append([slot, value])
            else:
                pred_belief_state.append([slot, np.random.choice(sample_values)])
        
        # Create instances
        gt_instance = {
            'dialogue_id': f'test_dialogue_{i // 5}',  # 5 turns per dialogue
            'turn_id': i % 5,
            'belief_state': gt_belief_state,
            'is_last_turn': (i % 5) == 4
        }
        
        pred_instance = {
            'dialogue_id': f'test_dialogue_{i // 5}',
            'turn_id': i % 5,
            'belief_state': pred_belief_state,
            'is_last_turn': (i % 5) == 4
        }
        
        predictions.append(pred_instance)
        ground_truth.append(gt_instance)
    
    return predictions, ground_truth


def show_data_sample():
    """Show sample data from the processed dataset"""
    
    print("\n" + "=" * 60)
    print("Data Sample Preview")
    print("=" * 60)
    
    # Load and show train data sample
    train_path = "data/processed/train_instances.json"
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        
        # Show first instance
        sample = train_data[0]
        print("\nSample Training Instance:")
        print(f"Dialogue ID: {sample['dialogue_id']}")
        print(f"Turn ID: {sample['turn_id']}")
        print(f"User Utterance: {sample['user_utterance']}")
        print(f"System Response: {sample.get('system_response', 'N/A')}")
        print(f"Domains: {sample.get('domains', [])}")
        print(f"Belief State: {sample['belief_state'][:3]}...")  # Show first 3 items
        print(f"Is Last Turn: {sample.get('is_last_turn', False)}")
        
        # Show dialogue history preview
        if sample.get('dialogue_history'):
            history_preview = sample['dialogue_history'][:200] + "..." if len(sample['dialogue_history']) > 200 else sample['dialogue_history']
            print(f"Dialogue History: {history_preview}")
    
    # Load and show statistics
    stats_path = "data/processed/statistics.json"
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"\nDataset Statistics:")
        print(f"Total Instances: {stats['total_instances']}")
        print(f"Number of Slots: {stats['num_slots']}")
        print(f"Domains: {stats['domains']}")
        print(f"Split Distribution:")
        for split, count in stats['splits'].items():
            percentage = (count / stats['total_instances']) * 100
            print(f"  {split}: {count} ({percentage:.1f}%)")


def check_requirements():
    """Check if all required files exist"""
    
    print("Checking requirements...")
    
    required_files = [
        "data/processed/train_instances.json",
        "data/processed/dev_instances.json", 
        "data/processed/test_instances.json",
        "data/processed/slot_meta.json",
        "data/processed/statistics.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some required files are missing. Please run:")
        print("   python src/data/preprocess.py")
        return False
    
    return True


def main():
    """Main test function"""
    
    print("DST Data Pipeline Test Suite")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Show data sample
    show_data_sample()
    
    # Run pipeline tests
    success = test_data_pipeline()
    
    if success:
        print("\n🚀 Ready to train DST models!")
        print("\nNext steps:")
        print("1. Implement your DST model")
        print("2. Use DSTDataLoader for training data")
        print("3. Use DSTEvaluator for evaluation")
        print("4. Save predictions and run evaluation")
    else:
        print("\n❌ Pipeline test failed. Please check the errors above.")


if __name__ == "__main__":
    main()