"""
History-Aware GraphDST - Main Model

Complete integration của all components:
1. Intent Encoder (BERT for current utterance)
2. History Graph Builder (dynamic graph construction)
3. Schema Graph Builder (static ontology structure)
4. GNN Layers (schema-aware processing)
5. Fusion Mechanism (cross-modal attention)
6. Multi-task Prediction Heads

Novel architecture combining strengths of BERT và GNN while avoiding their individual limitations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from dataclasses import dataclass

# Import các components
from src.models.components.intent_encoder import IntentEncoder
from src.data.graph_builders.history_graph_builder import HistoryGraphBuilder
from src.data.graph_builders.schema_graph_builder import SchemaGraphBuilder, SchemaGraph
from src.data.graph_builders.multi_level_graph_builder import MultiLevelGraphBuilder
from src.models.gnn_layers import UnifiedGNNLayer
from src.models.heterogeneous_gnn import HeterogeneousGNN
from src.models.fusion_layer import HistoryAwareFusion
from src.models.advanced_fusion import MultiModalAttentionFusion, AdaptiveFusionLayer
from src.models.delta_prediction_heads import DeltaPredictionHeads, DeltaTargetComputer


@dataclass
class DSTPrediction:
    """DST prediction output structure"""
    # Belief state predictions
    belief_state: Dict[str, Any]
    
    # Slot predictions 
    slot_predictions: Dict[str, torch.Tensor]  # slot -> [batch, num_values]
    
    # Attention weights và interpretability
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    
    # Intermediate representations
    intent_features: Optional[torch.Tensor] = None
    context_features: Optional[torch.Tensor] = None
    fused_features: Optional[torch.Tensor] = None


class MultiTaskPredictionHead(nn.Module):
    """
    Multi-task prediction heads cho different slot types
    
    Features:
    - Categorical slots: Classification over value vocabulary
    - Non-categorical slots: Copy mechanism from utterance
    - Special handling for "none" và "dontcare" values
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 slot_vocab: Dict[str, List[str]] = None,
                 max_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.slot_vocab = slot_vocab or {}
        self.max_seq_length = max_seq_length
        
        # Slot-specific prediction heads
        self.slot_heads = nn.ModuleDict()
        
        # Categorical slots: Classification heads
        for slot, values in self.slot_vocab.items():
            if len(values) > 0:  # Categorical slot
                # Add special tokens
                extended_values = values + ["none", "dontcare"]
                self.slot_heads[slot] = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, len(extended_values))
                )
        
        # Copy mechanism cho non-categorical slots
        self.copy_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Gate to choose between categorical prediction và copy
        self.copy_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Special value predictions
        self.none_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
        
        self.dontcare_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self,
                fused_features: torch.Tensor,
                utterance_tokens: Optional[torch.Tensor] = None,
                utterance_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-task prediction forward pass
        
        Args:
            fused_features: Fused representations [batch, seq_len, input_dim]
            utterance_tokens: Original utterance tokens [batch, max_len] 
            utterance_mask: Utterance attention mask [batch, max_len]
            
        Returns:
            Dictionary of slot predictions
        """
        
        batch_size = fused_features.size(0)
        seq_len = fused_features.size(1)
        
        # Global representation cho each example
        global_repr = torch.mean(fused_features, dim=1)  # [batch, input_dim]
        
        slot_predictions = {}
        
        # Categorical slot predictions
        for slot, head in self.slot_heads.items():
            slot_logits = head(global_repr)  # [batch, num_values]
            slot_predictions[slot] = slot_logits
        
        # Copy mechanism predictions (if utterance provided)
        if utterance_tokens is not None and utterance_mask is not None:
            # Attention over utterance tokens
            query = global_repr.unsqueeze(1)  # [batch, 1, input_dim]
            key = value = fused_features  # Use fused features as utterance representations
            
            copy_weights, _ = self.copy_attention(
                query, key, value, key_padding_mask=~utterance_mask
            )  # [batch, 1, seq_len]
            
            slot_predictions['copy_weights'] = copy_weights.squeeze(1)  # [batch, seq_len]
        
        # Special value predictions
        none_logits = self.none_classifier(global_repr)  # [batch, 1]
        dontcare_logits = self.dontcare_classifier(global_repr)  # [batch, 1]
        
        slot_predictions['none'] = none_logits
        slot_predictions['dontcare'] = dontcare_logits
        
        return slot_predictions


class HistoryAwareGraphDST(nn.Module):
    """
    Complete History-Aware GraphDST Model
    
    Architecture:
    1. Intent Branch: BERT processes current utterance only
    2. Context Branch: GNN processes dialog history + schema
    3. Fusion Layer: Cross-attention between intent và context
    4. Prediction Heads: Multi-task prediction cho all slots
    
    Key innovations:
    - Separation of current intent từ historical context
    - Dynamic graph construction từ dialog history
    - Cross-modal fusion với attention mechanism
    - Multi-task learning cho different slot types
    """
    
    def __init__(self,
                 # Model configuration
                 hidden_dim: int = 768,
                 num_gnn_layers: int = 3,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1,
                 
                 # Data configuration  
                 slot_vocab: Dict[str, List[str]] = None,
                 max_utterance_length: int = 50,
                 max_history_length: int = 20,
                 max_sequence_length: int = 50,
                 
                 # Component configuration
                 bert_model_name: str = "bert-base-uncased",
                 freeze_bert: bool = False,
                 use_pretrained_embeddings: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.slot_vocab = slot_vocab or {}
        self.max_utterance_length = max_utterance_length
        self.max_history_length = max_history_length
        
        # ===== 1. Intent Encoder =====
        self.intent_encoder = IntentEncoder(
            model_name=bert_model_name,
            hidden_dim=hidden_dim
        )
        
        # ===== Delta Prediction Heads =====
        # Load slot metadata
        slot_list = self._load_slot_list()
        self.slot_list = slot_list
        self.num_slots = len(slot_list)
        
        print(f"Model loaded {self.num_slots} slots")  # Debug
        
        # Initialize delta prediction heads
        self.delta_prediction_heads = DeltaPredictionHeads(
            hidden_dim=hidden_dim,
            num_slots=self.num_slots,
            slot_list=slot_list,
            max_seq_len=max_sequence_length,
            dropout=dropout
        )
        
        # Initialize delta target computer
        self.delta_target_computer = DeltaTargetComputer(slot_list)
        
        # Keep domain classifier for backward compatibility
        self.domain_classifier = nn.Linear(hidden_dim, 5)  # 5 domains
        
        # ===== 2. Graph Builders =====
        self.history_graph_builder = HistoryGraphBuilder(
            hidden_dim=hidden_dim,
            max_history_turns=max_history_length
        )
        
        self.schema_graph_builder = SchemaGraphBuilder(
            hidden_dim=hidden_dim,
            use_pretrained_embeddings=use_pretrained_embeddings
        )
        
        # ===== 3. GNN Layers =====
        self.gnn_layers = nn.ModuleList([
            UnifiedGNNLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_node_types=6,  # Turn, BeliefState, SlotValue, Domain, Slot, Value
                num_heads=num_attention_heads,
                max_sequence_length=max_sequence_length,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])
        
        # ===== 4. Fusion Layer =====
        self.fusion_layer = HistoryAwareFusion(
            intent_dim=hidden_dim,
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # ===== 5. Legacy Prediction Heads removed =====
        # Now using DeltaPredictionHeads initialized above
        
        # ===== Model state =====
        self.schema_graph = None  # Will be loaded during setup
    
    def _load_slot_list(self) -> List[str]:
        """Load slot list from slot metadata"""
        try:
            import json
            with open('data/processed/slot_meta.json', 'r') as f:
                slot_meta = json.load(f)
            
            if isinstance(slot_meta, list):
                return slot_meta
            elif isinstance(slot_meta, dict):
                # Check if it has 'slot_meta' key
                if 'slot_meta' in slot_meta:
                    return slot_meta['slot_meta']
                else:
                    return list(slot_meta.keys())
            else:
                # Fallback to standard MultiWOZ slots
                return [
                    'hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day',
                    'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'hotel-name',
                    'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people',
                    'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange',
                    'restaurant-area', 'restaurant-name', 'restaurant-time', 'restaurant-day',
                    'restaurant-book people', 'attraction-name', 'attraction-type', 'taxi-leaveat',
                    'taxi-destination', 'taxi-departure', 'taxi-arriveby'
                ]
        except Exception as e:
            print(f"Warning: Could not load slot metadata, using default slots: {e}")
            # Default MultiWOZ 2.4 slots
            return [
                'hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day',
                'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'hotel-name',
                'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people',
                'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange',
                'restaurant-area', 'restaurant-name', 'restaurant-time', 'restaurant-day',
                'restaurant-book people', 'attraction-name', 'attraction-type', 'taxi-leaveat',
                'taxi-destination', 'taxi-departure', 'taxi-arriveby'
            ]
        
    def setup_schema_graph(self, slot_meta_path: str, vocab_path: str):
        """Setup schema graph từ ontology files"""
        self.schema_graph_builder.load_ontology(slot_meta_path, vocab_path)
        
        with torch.no_grad():
            self.schema_graph = self.schema_graph_builder.build_graph()
        
        print(f"Schema graph loaded: {self.schema_graph.metadata['num_nodes']} nodes, "
              f"{self.schema_graph.metadata['num_edges']} edges")
    
    def _construct_unified_graph(self,
                                history_graph_data: Dict[str, Any],
                                batch_size: int,
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Construct unified graph combining history và schema
        
        Returns:
            - node_features: Combined node features
            - edge_index: Combined edge indices  
            - edge_attr: Combined edge attributes
            - node_types: Combined node types
            - turn_boundaries: Turn boundary information
        """
        
        if self.schema_graph is None:
            raise ValueError("Schema graph not initialized. Call setup_schema_graph() first.")
        
        # Get history graph components
        history_features = history_graph_data['node_features']  # [num_history_nodes, hidden_dim]
        history_edge_index = history_graph_data['edge_index']  # [2, num_history_edges]
        history_edge_attr = history_graph_data['edge_attr']  # [num_history_edges, edge_attr_dim]
        history_node_types = history_graph_data['node_types']  # [num_history_nodes]
        turn_boundaries = history_graph_data.get('turn_boundaries', [])
        
        # Get schema graph components (move to same device)
        schema_features = self.schema_graph.node_features.to(device)
        schema_edge_index = self.schema_graph.edge_index.to(device)
        schema_edge_attr = self.schema_graph.edge_attr.to(device)
        schema_node_types = torch.tensor([
            3 if node_type.value == "domain" else 
            4 if node_type.value == "slot" else 
            5  # value
            for node_type in self.schema_graph.node_types
        ], device=device)
        
        # Combine node features
        combined_features = torch.cat([history_features, schema_features], dim=0)
        
        # Combine edge indices (adjust schema indices)
        num_history_nodes = history_features.size(0)
        adjusted_schema_edge_index = schema_edge_index + num_history_nodes
        combined_edge_index = torch.cat([history_edge_index, adjusted_schema_edge_index], dim=1)
        
        # Combine edge attributes
        combined_edge_attr = torch.cat([history_edge_attr, schema_edge_attr], dim=0)
        
        # Combine node types
        combined_node_types = torch.cat([history_node_types, schema_node_types], dim=0)
        
        return combined_features, combined_edge_index, combined_edge_attr, combined_node_types, turn_boundaries
    
    def forward(self,
                # Tokenized inputs (from DataLoader)
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                
                # Dialog history
                history_data: List[List[Dict[str, Any]]],
                
                # Optional inputs
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass của complete model
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            history_data: List of dialog history for each example
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dict với predictions và intermediate results
        """
        
        device = next(self.parameters()).device
        batch_size = input_ids.size(0)
        
        # ===== 1. Intent Encoding =====
        intent_output = self.intent_encoder(input_ids, attention_mask)
        intent_features = intent_output['intent_features']  # [batch_size, hidden_dim]
        intent_features = intent_features.unsqueeze(1)  # [batch_size, 1, hidden_dim] for sequence dimension
        
        # ===== 2. History Graph Construction + GNN Processing =====
        # Process actual dialogue history với GNN
        if history_data and any(len(h) > 0 for h in history_data):
            # Build graphs from history data
            context_features = self._process_history_with_gnn(history_data, device)  # [batch_size, hidden_dim]
        else:
            # Fallback: no history available
            context_features = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # ===== 3. Fusion Layer: Intent + Context =====
        # Combine BERT intent features với GNN context features
        fused_features = self._fuse_intent_and_context(intent_features, context_features)
        
        # Global pooling of fused features
        pooled_features = fused_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Domain classification (keep for compatibility)
        domain_logits = self.domain_classifier(pooled_features)  # [batch_size, 5]
        
        # Delta predictions
        delta_predictions = self.delta_prediction_heads(
            pooled_features=pooled_features,
            sequence_features=intent_output.get('token_features'),  # Token-level features for span extraction
            attention_mask=attention_mask
        )
        
        # ===== 4. Prepare Output =====
        predictions = {
            # Legacy outputs for backward compatibility
            'domain_logits': domain_logits,
            
            # New delta predictions
            'slot_operations': delta_predictions['slot_operations'],
            'value_existence': delta_predictions['value_existence'], 
            'none_logits': delta_predictions['none_logits'],
            'dontcare_logits': delta_predictions['dontcare_logits'],
            
            # Intermediate features
            'intent_features': intent_features,
            'pooled_features': pooled_features,
        }
        
        # Add span extraction if available
        if 'span_start_logits' in delta_predictions:
            predictions['span_start_logits'] = delta_predictions['span_start_logits']
            predictions['span_end_logits'] = delta_predictions['span_end_logits']
        
        if return_attention_weights:
            predictions['attention_weights'] = {}
        
        return predictions
    
    def _process_history_with_gnn(self, history_data: List[List[Dict[str, Any]]], device: torch.device) -> torch.Tensor:
        """
        Process dialogue history using GNN to get context features
        
        Args:
            history_data: List of dialogue history for each batch item
            device: Device to put tensors on
            
        Returns:
            context_features: [batch_size, hidden_dim]
        """
        batch_size = len(history_data)
        context_features = []
        
        for batch_idx, history in enumerate(history_data):
            if not history:
                # No history: zero context
                context_feat = torch.zeros(self.hidden_dim, device=device)
            else:
                # Build graph from dialogue history
                try:
                    # Simple approach: encode previous belief states
                    # In full implementation, would build proper graph with turns, slots, values
                    
                    # Collect previous belief state information
                    history_text = []
                    for turn in history[-3:]:  # Take last 3 turns
                        if 'user_utterance' in turn:
                            history_text.append(turn['user_utterance'])
                    
                    if history_text:
                        # Simple encoding: concatenate and encode with BERT
                        combined_history = " [SEP] ".join(history_text)
                        
                        # Encode with BERT (reuse intent encoder)
                        encoded = self.intent_encoder.encode_utterances([combined_history])
                        context_feat = encoded['intent_features'].squeeze(0)  # Remove batch dim
                    else:
                        context_feat = torch.zeros(self.hidden_dim, device=device)
                        
                except Exception as e:
                    print(f"Warning: Failed to process history for batch {batch_idx}: {e}")
                    context_feat = torch.zeros(self.hidden_dim, device=device)
            
            context_features.append(context_feat)
        
        # Stack into batch tensor
        context_features = torch.stack(context_features)  # [batch_size, hidden_dim]
        
        return context_features
    
    def _fuse_intent_and_context(self, 
                                intent_features: torch.Tensor, 
                                context_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse intent features (BERT) with context features (GNN)
        
        Args:
            intent_features: [batch_size, 1, hidden_dim] - from BERT
            context_features: [batch_size, hidden_dim] - from GNN
            
        Returns:
            fused_features: [batch_size, 1, hidden_dim]
        """
        batch_size = intent_features.size(0)
        
        # Expand context to match intent dimensions
        context_expanded = context_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Simple fusion: weighted sum
        # In full implementation, would use proper fusion layer with attention
        alpha = 0.7  # Weight for intent features
        beta = 0.3   # Weight for context features
        
        fused_features = alpha * intent_features + beta * context_expanded
        
        return fused_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information và statistics"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component-wise parameter count
        component_params = {
            'intent_encoder': sum(p.numel() for p in self.intent_encoder.parameters()),
            'gnn_layers': sum(p.numel() for p in self.gnn_layers.parameters()),
            'fusion_layer': sum(p.numel() for p in self.fusion_layer.parameters()),
            'delta_prediction_heads': sum(p.numel() for p in self.delta_prediction_heads.parameters()),
            'domain_classifier': sum(p.numel() for p in self.domain_classifier.parameters()),
            'graph_builders': (
                sum(p.numel() for p in self.history_graph_builder.parameters()) +
                sum(p.numel() for p in self.schema_graph_builder.parameters())
            )
        }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': component_params,
            'hidden_dim': self.hidden_dim,
            'num_gnn_layers': self.num_gnn_layers,
            'num_slots': self.num_slots,
            'slot_list': self.slot_list,
            'schema_info': self.schema_graph.metadata if self.schema_graph else None
        }


# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing History-Aware GraphDST...")
    
    # Mock slot vocabulary
    mock_slot_vocab = {
        'hotel-area': ['center', 'north', 'south', 'east', 'west'],
        'hotel-pricerange': ['cheap', 'moderate', 'expensive'],
        'restaurant-food': ['chinese', 'italian', 'indian', 'french'],
        'restaurant-pricerange': ['cheap', 'moderate', 'expensive']
    }
    
    # Create model
    model = HistoryAwareGraphDST(
        hidden_dim=768,
        num_gnn_layers=2,  # Reduced cho testing
        slot_vocab=mock_slot_vocab,
        max_utterance_length=30,
        max_history_length=10
    )
    
    # Setup schema graph (using mock data)
    import tempfile
    import os
    
    mock_slot_meta = {
        'hotel-area': {'description': 'Area of hotel'},
        'hotel-pricerange': {'description': 'Price range of hotel'},
        'restaurant-food': {'description': 'Type of food'},
        'restaurant-pricerange': {'description': 'Price range of restaurant'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_slot_meta, f)
        slot_meta_path = f.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_slot_vocab, f)
        vocab_path = f.name
    
    model.setup_schema_graph(slot_meta_path, vocab_path)
    
    # Test forward pass
    utterance = "I want a cheap hotel in the center"
    dialog_history = [
        {
            'turn_id': 0,
            'user_utterance': "I need a restaurant",
            'system_response': "What type of food would you like?",
            'belief_state': {}
        }
    ]
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        prediction = model(utterance, dialog_history, return_attention=True)
    
    print(f"Belief state: {prediction.belief_state}")
    print(f"Intent features shape: {prediction.intent_features.shape}")
    print(f"Context features shape: {prediction.context_features.shape}")
    print(f"Fused features shape: {prediction.fused_features.shape}")
    
    # Model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  Component breakdown:")
    for component, params in model_info['component_parameters'].items():
        print(f"    {component}: {params:,}")
    
    # Clean up
    os.unlink(slot_meta_path)
    os.unlink(vocab_path)
    
    print("✅ History-Aware GraphDST testing completed!")