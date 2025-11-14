"""
Updated History-Aware GraphDST - Main Model với Multi-level Architecture

Theo yêu cầu:
1. Current utterance → BERT
2. Multi-level Graph Structure: Domain, Schema, Value Graph với Heterogeneous GNN + Temporal modeling  
3. Kết hợp (1) + (2) → Delta prediction
4. Classification approach cho tất cả slots

Author: Assistant
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
from dataclasses import dataclass

# Import các components mới
from src.models.components.intent_encoder import IntentEncoder
from src.data.graph_builders.multi_level_graph_builder import MultiLevelGraphBuilder  
from src.models.heterogeneous_gnn import HeterogeneousGNN
from src.models.advanced_fusion import MultiModalAttentionFusion, AdaptiveFusionLayer
from src.models.classification_delta_heads import ClassificationDeltaPredictionHeads, ClassificationDeltaTargetComputer


@dataclass  
class DSTPrediction:
    """DST prediction output structure"""
    # Belief state predictions
    belief_state: Dict[str, Any]
    
    # Slot predictions 
    slot_predictions: Dict[str, torch.Tensor]
    
    # Delta operations
    delta_operations: Optional[Dict[str, torch.Tensor]] = None
    
    # Attention weights và interpretability
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    
    # Intermediate representations
    intent_features: Optional[torch.Tensor] = None
    context_features: Optional[torch.Tensor] = None
    fused_features: Optional[torch.Tensor] = None


class UpdatedHistoryAwareGraphDST(nn.Module):
    """
    Updated History-Aware GraphDST với Multi-level Graph Architecture
    
    Architecture Flow:
    1. Current utterance → BERT Intent Encoder
    2. Dialogue history → Multi-level Graph Builder → Heterogeneous GNN  
    3. Intent + Graph features → Advanced Fusion
    4. Fused features → Delta Prediction (KEEP/ADD/UPDATE/REMOVE)
    """
    
    def __init__(self,
                 # Model dimensions
                 hidden_dim: int = 768,
                 fusion_dim: int = 768,
                 
                 # Multi-level graph config
                 num_domains: int = 5,
                 max_history_turns: int = 10,
                 
                 # GNN config
                 num_gnn_layers: int = 3,
                 num_heads: int = 8,
                 
                 # Training config
                 dropout: float = 0.1,
                 
                 # Fusion type
                 fusion_type: str = 'multimodal',  # 'multimodal' or 'adaptive'
                 
                 # Slot vocabulary cho classification
                 slot_vocab: Optional[Dict[str, List[str]]] = None):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_domains = num_domains
        self.max_history_turns = max_history_turns
        self.num_gnn_layers = num_gnn_layers
        self.fusion_type = fusion_type
        self.dropout = dropout
        
        # ===== 1. Intent Encoder (BERT Branch) =====
        self.intent_encoder = IntentEncoder(
            model_name='bert-base-uncased',
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # ===== 2. Multi-level Graph Builder =====
        self.multi_level_graph_builder = MultiLevelGraphBuilder(
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            max_history_turns=max_history_turns,
            dropout=dropout
        )
        
        # ===== 3. Heterogeneous GNN =====
        self.heterogeneous_gnn = HeterogeneousGNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_node_types=4,  # domain, slot, value, turn
            num_heads=num_heads,
            dropout=dropout,
            temporal_layers=1
        )
        
        # ===== 4. Advanced Fusion Layer =====
        if fusion_type == 'multimodal':
            self.fusion_layer = MultiModalAttentionFusion(
                intent_dim=hidden_dim,
                graph_dim=hidden_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gate=True
            )
        elif fusion_type == 'adaptive':
            self.fusion_layer = AdaptiveFusionLayer(
                intent_dim=hidden_dim,
                graph_dim=hidden_dim,
                fusion_dim=fusion_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # ===== 5. Pure Classification Delta Prediction Heads =====
        # Load slots and slot value vocabulary
        self.slots = self._load_slot_list()
        self.slot_value_vocab = self._load_slot_value_vocab(self.slots)
        
        self.delta_target_computer = ClassificationDeltaTargetComputer(
            slot_list=self.slots,
            slot_value_vocab=self.slot_value_vocab
        )
        
        self.delta_prediction_heads = ClassificationDeltaPredictionHeads(
            hidden_dim=fusion_dim,
            num_slots=len(self.slots),
            slot_list=self.slots,
            slot_value_vocab=self.slot_value_vocab,
            dropout=dropout
        )
        
        # ===== Model State =====
        self.ontology_loaded = False
        
        print(f"✅ Updated HistoryAwareGraphDST initialized:")
        print(f"   - Intent encoder: {hidden_dim}D BERT")
        print(f"   - Multi-level graph: {num_domains} domains, {max_history_turns} turns")
        print(f"   - Heterogeneous GNN: {num_gnn_layers} layers, {num_heads} heads")
        print(f"   - Fusion: {fusion_type}")
        print(f"   - Slots: {len(self.slots)}")
    
    def _load_slot_list(self) -> List[str]:
        """Load slot list from slot metadata"""
        try:
            candidate_paths = [
                'data/processed_graph/slot_meta.json',
                'data/processed/slot_meta.json'
            ]
            slot_meta = None
            for path in candidate_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        slot_meta = json.load(f)
                    print(f"Loaded slot metadata from {path}")
                    break
            if slot_meta is None:
                raise FileNotFoundError("Slot metadata not found in default locations")
            
            if isinstance(slot_meta, list):
                return slot_meta
            elif isinstance(slot_meta, dict) and 'slot_meta' in slot_meta:
                return slot_meta['slot_meta']
            else:
                return list(slot_meta.keys())
        except Exception as e:
            print(f"Warning: Could not load slot metadata: {e}")
            # Default MultiWOZ slots
            return [
                'hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day',
                'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'hotel-name',
                'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people', 
                'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange',
                'restaurant-area', 'restaurant-name', 'restaurant-book time', 'restaurant-book day',
                'restaurant-book people', 'attraction-name', 'attraction-type', 'taxi-leaveat',
                'taxi-destination', 'taxi-departure', 'taxi-arriveby'
            ]
    
    def _load_slot_value_vocab(self, slot_list: List[str]) -> Dict[str, List[str]]:
        """Initialize slot value vocabulary container"""
        return {slot: [] for slot in slot_list}
    
    def setup_ontology(self, 
                       slot_meta_path: str, 
                       ontology_path: str,
                       slot_value_vocab: Optional[Dict[str, List[str]]] = None):
        """Setup ontology cho multi-level graph builder"""
        print("Setting up ontology for multi-level graph...")
        
        # Load slot meta vào model
        with open(slot_meta_path, 'r') as f:
            slot_meta = json.load(f)
        
        if isinstance(slot_meta, list):
            self.slots = slot_meta
        else:
            raise ValueError(f"slot_meta must be a list, got {type(slot_meta)}")
        
        print(f"  Loaded {len(self.slots)} slots from {slot_meta_path}")
        
        # Update slot value vocabulary if provided
        if slot_value_vocab is not None:
            processed_vocab = {}
            for slot in self.slots:
                values = slot_value_vocab.get(slot, [])
                if isinstance(values, set):
                    values = sorted(values)
                elif isinstance(values, list):
                    values = sorted(dict.fromkeys(values))
                else:
                    values = []
                processed_vocab[slot] = values
            self.slot_value_vocab = processed_vocab
        else:
            # Ensure keys exist for all slots
            self.slot_value_vocab = {
                slot: self.slot_value_vocab.get(slot, [])
                for slot in self.slots
            }
        
        # Rebuild delta prediction heads với đúng số slots
        print(f"  Rebuilding delta prediction heads...")
        self.delta_prediction_heads = ClassificationDeltaPredictionHeads(
            hidden_dim=self.fusion_dim,
            num_slots=len(self.slots),
            slot_list=self.slots,
            slot_value_vocab=self.slot_value_vocab,
            dropout=self.dropout
        )
        self.delta_target_computer = ClassificationDeltaTargetComputer(
            slot_list=self.slots,
            slot_value_vocab=self.slot_value_vocab
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        self.delta_prediction_heads = self.delta_prediction_heads.to(device)
        
        # Setup graph builder
        self.multi_level_graph_builder.load_ontology(slot_meta_path, ontology_path)
        self.ontology_loaded = True
        
        print("✅ Ontology loaded successfully")
    
    def _process_intent_with_bert(self, utterance: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process current utterance với BERT Intent Encoder
        
        Args:
            utterance: Current user utterance
            
        Returns:
            intent_features: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] 
        """
        # Encode utterance
        encoding_result = self.intent_encoder.encode_utterance(utterance)
        
        intent_features = encoding_result['token_embeddings']  # [1, seq_len, hidden_dim]
        attention_mask = encoding_result.get('attention_mask', None)  # [1, seq_len]
        
        # Move to correct device
        device = next(self.parameters()).device
        intent_features = intent_features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        return intent_features, attention_mask
    
    def _process_history_with_multilevel_gnn(self, 
                                           dialogue_history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process dialogue history với Multi-level Graph + Heterogeneous GNN
        
        Args:
            dialogue_history: List of previous turns
            
        Returns:
            Dictionary containing graph processing results
        """
        if not self.ontology_loaded:
            raise ValueError("Ontology not loaded. Call setup_ontology() first.")
        
        # ===== Step 1: Build Multi-level Graph =====
        graph_data = self.multi_level_graph_builder(dialogue_history)
        
        # Move graph data to correct device
        device = next(self.parameters()).device
        for key in graph_data:
            if isinstance(graph_data[key], torch.Tensor):
                graph_data[key] = graph_data[key].to(device)
        
        # ===== Step 2: Process với Heterogeneous GNN =====
        gnn_output = self.heterogeneous_gnn(graph_data, return_attention=True)
        
        return {
            'graph_features': gnn_output['graph_features'],              # [1, hidden_dim]
            'node_features': gnn_output['node_features'],               # [num_nodes, hidden_dim]
            'graph_representation': gnn_output['graph_representation'], # [1, hidden_dim]
            'temporal_representation': gnn_output['temporal_representation'], # [1, hidden_dim]
            'attention_weights': gnn_output.get('attention_weights', {}),
            'raw_graph_data': graph_data
        }
    
    def _fuse_intent_and_context(self,
                                intent_features: torch.Tensor,
                                graph_features: torch.Tensor,
                                intent_mask: Optional[torch.Tensor] = None,
                                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Fuse intent và graph features với Advanced Fusion
        
        Args:
            intent_features: [batch, seq_len, hidden_dim] từ BERT
            graph_features: [batch, 1, hidden_dim] từ GNN
            intent_mask: [batch, seq_len] attention mask
            return_attention: Return attention weights
            
        Returns:
            Fusion results dictionary
        """
        if self.fusion_type == 'multimodal':
            fusion_output = self.fusion_layer(
                intent_features=intent_features,
                graph_features=graph_features,
                intent_mask=intent_mask,
                return_attention=return_attention
            )
        elif self.fusion_type == 'adaptive':
            fusion_output = self.fusion_layer(
                intent_features=intent_features,
                graph_features=graph_features
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fusion_output
    
    def forward(self,
                # Required inputs
                utterance: str,
                dialogue_history: List[Dict[str, Any]],
                
                # Optional previous state for delta computation
                previous_belief_state: Optional[Dict[str, str]] = None,
                
                # Control flags
                return_attention: bool = False) -> DSTPrediction:
        """
        Main forward pass - FIXED: No data leakage
        
        Args:
            utterance: Current user utterance
            dialogue_history: Previous turns with belief states
            previous_belief_state: Previous turn belief state (for incremental update)
            return_attention: Return attention weights
            
        NOTE: current_belief_state REMOVED to prevent data leakage!
              Targets should be computed outside the model.
              
        Returns:
            DSTPrediction containing results
        """
        
        # ===== 1. Process Intent với BERT =====
        intent_features, intent_mask = self._process_intent_with_bert(utterance)
        # intent_features: [1, seq_len, hidden_dim]
        
        # ===== 2. Process History với Multi-level GNN =====
        gnn_output = self._process_history_with_multilevel_gnn(dialogue_history)
        graph_features = gnn_output['graph_features']  # [1, hidden_dim]
        
        # Expand graph_features to match fusion input format
        graph_features = graph_features.unsqueeze(1)  # [1, 1, hidden_dim]
        
        # ===== 3. Fusion Intent + Graph =====
        fusion_output = self._fuse_intent_and_context(
            intent_features=intent_features,
            graph_features=graph_features,
            intent_mask=intent_mask,
            return_attention=return_attention
        )
        
        fused_features = fusion_output['fused_features']  # [1, 1, fusion_dim]
        
        # ===== 4. Pure Classification Delta Prediction =====
        
        # REMOVED: Delta target computation (data leakage fix)
        # Targets should be computed outside model in training loop
        
        # Extract pooled features từ fused_features
        pooled_features = fused_features.squeeze(1)  # [batch, fusion_dim]
        
        # Ensure all tensors on correct device
        device = next(self.parameters()).device
        pooled_features = pooled_features.to(device)
        
        # Get pure classification predictions
        delta_predictions = self.delta_prediction_heads(
            pooled_features=pooled_features
        )
        
        # ===== 5. Construct Belief State from Classification Predictions =====
        
        # Extract operations
        slot_operations = delta_predictions['slot_operations']  # [1, num_slots, 4] 
        
        # Get value predictions
        value_logits = delta_predictions.get('values', {})
        none_logits = delta_predictions.get('none')        # [1, num_slots]
        dontcare_logits = delta_predictions.get('dontcare')  # [1, num_slots]
        
        # Predict belief state dựa trên delta operations + value classification
        predicted_belief_state = {}
        
        for i, slot in enumerate(self.slots):
            # Get operation prediction
            operation_logits = slot_operations[0, i, :]  # [4]  
            predicted_operation = torch.argmax(operation_logits).item()
            
            # Apply operation logic
            if predicted_operation == 0:  # KEEP
                if previous_belief_state and slot in previous_belief_state:
                    predicted_belief_state[slot] = previous_belief_state[slot]
                    
            elif predicted_operation in [1, 2]:  # ADD or UPDATE
                # Check special values first
                none_score = torch.sigmoid(none_logits[0, i]).item() if none_logits is not None else 0.0
                dontcare_score = torch.sigmoid(dontcare_logits[0, i]).item() if dontcare_logits is not None else 0.0
                
                if none_score > 0.5:
                    predicted_belief_state[slot] = "none"
                elif dontcare_score > 0.5:
                    predicted_belief_state[slot] = "dontcare"
                elif slot in value_logits:
                    # Classification approach
                    logits = value_logits[slot][0]  # [vocab_size]
                    pred_value = self.delta_prediction_heads.get_value_from_logits(
                        slot, logits
                    )
                    predicted_belief_state[slot] = pred_value
                    
            elif predicted_operation == 3:  # REMOVE
                # Do not add to belief state (effectively removes)
                pass
        
        # ===== 6. Prepare Output =====
        
        # Collect attention weights
        all_attention_weights = {}
        if return_attention:
            all_attention_weights.update(gnn_output.get('attention_weights', {}))
            all_attention_weights.update(fusion_output.get('attention_weights', {}))
        
        return DSTPrediction(
            belief_state=predicted_belief_state,
            slot_predictions=delta_predictions,
            delta_operations=delta_predictions.get('slot_operations'),
            attention_weights=all_attention_weights if return_attention else None,
            intent_features=intent_features,
            context_features=gnn_output['graph_representation'],
            fused_features=fused_features
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            'model_name': 'UpdatedHistoryAwareGraphDST',
            'architecture': {
                '1_intent_encoder': 'BERT-based Intent Encoder',
                '2_graph_builder': 'Multi-level Graph Builder (Domain/Schema/Value/Turn)',
                '3_heterogeneous_gnn': f'{self.num_gnn_layers}-layer Heterogeneous GNN + Temporal GRU',
                '4_fusion': f'{self.fusion_type.capitalize()} Attention Fusion',
                '5_prediction': 'Delta Prediction Heads (KEEP/ADD/UPDATE/REMOVE)'
            },
            'parameters': {
                'hidden_dim': self.hidden_dim,
                'fusion_dim': self.fusion_dim,
                'num_domains': self.num_domains,
                'max_history_turns': self.max_history_turns,
                'num_gnn_layers': self.num_gnn_layers,
                'fusion_type': self.fusion_type,
                'num_slots': len(self.slots)
            },
            'features': [
                'Multi-level graph structure (Domain, Schema, Value, Turn)',
                'Heterogeneous GNN với specialized node types',
                'Temporal modeling với GRU + attention',
                'Cross-modal fusion với multi-head attention',
                'Delta-based state tracking (KEEP/ADD/UPDATE/REMOVE)',
                'Classification approach cho all slots'
            ]
        }


# Test function
def test_updated_model():
    """Test Updated HistoryAwareGraphDST"""
    print("Testing Updated History-Aware GraphDST...")
    
    # Initialize model
    model = UpdatedHistoryAwareGraphDST(
        hidden_dim=768,
        fusion_dim=768,
        num_domains=5,
        max_history_turns=5,
        num_gnn_layers=2,
        fusion_type='multimodal'
    )
    
    # Sample data
    utterance = "I want a cheap hotel in the center"
    dialogue_history = [
        {
            'transcript': 'Hello, I need help booking a hotel',
            'belief_state': [['hotel-area', 'centre']]
        }
    ]
    
    previous_belief_state = {'hotel-area': 'centre'}
    current_belief_state = {'hotel-area': 'centre', 'hotel-pricerange': 'cheap'}
    
    # Setup ontology (mock)
    try:
        model.setup_ontology('data/processed/slot_meta.json', 'data/raw/ontology.json')
    except:
        print("Note: Ontology files not found, using mock setup")
        model.ontology_loaded = True
    
    # Forward pass
    with torch.no_grad():
        try:
            prediction = model(
                utterance=utterance,
                dialogue_history=dialogue_history,
                previous_belief_state=previous_belief_state,
                current_belief_state=current_belief_state,
                return_attention=True
            )
            
            print(f"✅ Model forward pass successful!")
            print(f"   - Belief state: {len(prediction.belief_state)} slots")
            print(f"   - Slot predictions: {list(prediction.slot_predictions.keys())}")
            print(f"   - Delta operations shape: {prediction.delta_operations.shape if prediction.delta_operations is not None else 'None'}")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
    
    # Print model info
    info = model.get_model_info()
    print(f"\n📋 Model Architecture:")
    for key, value in info['architecture'].items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    test_updated_model()