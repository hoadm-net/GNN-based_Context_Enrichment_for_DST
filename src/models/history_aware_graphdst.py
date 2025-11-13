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
from src.models.gnn_layers import UnifiedGNNLayer
from src.models.fusion_layer import HistoryAwareFusion


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
            hidden_dim=hidden_dim,
            max_length=max_utterance_length,
            freeze_bert=freeze_bert,
            dropout=dropout
        )
        
        # ===== 2. Graph Builders =====
        self.history_graph_builder = HistoryGraphBuilder(
            hidden_dim=hidden_dim,
            max_history_length=max_history_length
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
        
        # ===== 5. Prediction Heads =====
        self.prediction_head = MultiTaskPredictionHead(
            input_dim=hidden_dim,
            slot_vocab=slot_vocab,
            dropout=dropout
        )
        
        # ===== Model state =====
        self.schema_graph = None  # Will be loaded during setup
        
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
                # Current utterance
                utterance: str,
                
                # Dialog history
                dialog_history: List[Dict[str, Any]],
                
                # Optional inputs
                utterance_tokens: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> DSTPrediction:
        """
        Forward pass của complete model
        
        Args:
            utterance: Current user utterance
            dialog_history: List of previous dialog turns
            utterance_tokens: Pre-tokenized utterance (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            DSTPrediction object với all predictions và intermediate results
        """
        
        device = next(self.parameters()).device
        batch_size = 1  # Currently support single example
        
        # ===== 1. Intent Encoding =====
        intent_output = self.intent_encoder([utterance])  # Pass as list for batch processing
        intent_features = intent_output['intent_features']  # [1, hidden_dim]
        intent_features = intent_features.unsqueeze(1)  # [1, 1, hidden_dim] for sequence dimension
        
        # ===== 2. History Graph Construction =====
        history_graph = self.history_graph_builder.build_graph(dialog_history)
        history_graph_data = history_graph.to_dict()  # Convert to dict format
        
        # Move history graph to device
        for key in ['node_features', 'edge_index', 'edge_attr', 'node_types']:
            if key in history_graph_data:
                history_graph_data[key] = history_graph_data[key].to(device)
        
        # ===== 3. Unified Graph Construction =====
        (combined_features, combined_edge_index, combined_edge_attr, 
         combined_node_types, turn_boundaries) = self._construct_unified_graph(
            history_graph_data, batch_size, device
        )
        
        # ===== 4. GNN Processing =====
        context_features = combined_features
        temporal_positions = torch.arange(context_features.size(0), device=device)
        
        for gnn_layer in self.gnn_layers:
            context_features = gnn_layer(
                x=context_features,
                edge_index=combined_edge_index,
                edge_attr=combined_edge_attr,
                node_types=combined_node_types,
                temporal_positions=temporal_positions,
                turn_boundaries=turn_boundaries
            )
        
        # Add batch dimension
        context_features = context_features.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # ===== 5. Cross-Modal Fusion =====
        fusion_output = self.fusion_layer(
            intent_features=intent_features,
            context_features=context_features,
            return_attention_weights=return_attention
        )
        
        fused_features = fusion_output['fused_features']
        enhanced_intent = fusion_output['enhanced_intent']
        enhanced_context = fusion_output['enhanced_context']
        
        # ===== 6. Multi-Task Prediction =====
        slot_predictions = self.prediction_head(
            fused_features=fused_features,
            utterance_tokens=utterance_tokens
        )
        
        # ===== 7. Construct Belief State =====
        belief_state = {}
        
        for slot, logits in slot_predictions.items():
            if slot in ['copy_weights', 'none', 'dontcare']:
                continue
                
            if slot in self.slot_vocab:
                # Categorical slot
                values = self.slot_vocab[slot] + ["none", "dontcare"]
                pred_idx = torch.argmax(logits, dim=-1).item()
                predicted_value = values[pred_idx]
                belief_state[slot] = predicted_value
        
        # ===== 8. Prepare Output =====
        attention_weights = None
        if return_attention:
            attention_weights = fusion_output.get('attention_weights', {})
        
        return DSTPrediction(
            belief_state=belief_state,
            slot_predictions=slot_predictions,
            attention_weights=attention_weights,
            intent_features=enhanced_intent,
            context_features=enhanced_context,
            fused_features=fused_features
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information và statistics"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component-wise parameter count
        component_params = {
            'intent_encoder': sum(p.numel() for p in self.intent_encoder.parameters()),
            'gnn_layers': sum(p.numel() for p in self.gnn_layers.parameters()),
            'fusion_layer': sum(p.numel() for p in self.fusion_layer.parameters()),
            'prediction_head': sum(p.numel() for p in self.prediction_head.parameters()),
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
            'num_slots': len(self.slot_vocab),
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