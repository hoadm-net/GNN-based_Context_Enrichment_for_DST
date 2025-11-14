"""
Batch-optimized Multi-Level GNN-based DST for Delta Prediction
Simpler architecture focused on training efficiency
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Any
import sys
sys.path.append('src')

from data.graph_builders.multi_level_graph_builder import MultiLevelGraphBuilder
from models.gnn_layers import TemporalReasoningGNN


class BatchGraphDST(nn.Module):
    """
    Simplified batch-optimized GNN-DST for delta prediction
    
    Architecture:
    1. BERT for current utterance encoding
    2. Multi-level Graph Builder for dialogue history
    3. Heterogeneous GNN for graph reasoning
    4. Fusion + Prediction heads for delta output
    """
    
    def __init__(self,
                 slot_meta: List[str],
                 value_vocab_size: int,
                 ontology: Dict,
                 bert_model: str = 'bert-base-uncased',
                 hidden_dim: int = 768,
                 gnn_layers: int = 2,
                 num_gate_classes: int = 4,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.slot_meta = slot_meta
        self.num_slots = len(slot_meta)
        self.value_vocab_size = value_vocab_size
        self.hidden_dim = hidden_dim
        self.num_gate_classes = num_gate_classes
        
        # BERT encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        
        # Multi-level graph builder
        self.graph_builder = MultiLevelGraphBuilder(
            hidden_dim=hidden_dim,
            num_domains=5,  # MultiWOZ domains
            max_history_turns=20,
            dropout=dropout
        )
        
        # Load ontology into graph builder
        self.graph_builder.load_ontology(slot_meta, ontology)
        
        # Temporal GNN
        self.gnn = TemporalReasoningGNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Slot gate predictor (none/span/dontcare/delete)
        self.gate_pred = nn.Linear(hidden_dim, num_gate_classes)
        
        # Value predictor (for slots with gate=span)
        self.value_pred = nn.Linear(hidden_dim, value_vocab_size)
        
        # Slot-specific linear layers
        self.slot_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_slots)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_utterance(self, utterances: List[str]) -> torch.Tensor:
        """
        Encode batch of utterances with BERT
        
        Returns:
            [batch_size, hidden_dim]
        """
        # Tokenize
        encoding = self.tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.bert.device)
        
        # BERT forward
        outputs = self.bert(**encoding)
        
        # Use [CLS] token
        cls_features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        
        return cls_features
    
    def process_graph_batch(self, dialogue_histories: List[List[Dict]]) -> torch.Tensor:
        """
        Process batch of dialogue histories through graph builder + GNN
        
        Args:
            dialogue_histories: List of dialogue history (each is List[Dict])
        
        Returns:
            graph_features: [batch_size, hidden_dim]
        """
        batch_size = len(dialogue_histories)
        graph_features_list = []
        
        # Process each instance separately (graph structure varies)
        for history in dialogue_histories:
            if len(history) == 0:
                # No history - use zero vector
                graph_feat = torch.zeros(1, self.hidden_dim).to(self.bert.device)
            else:
                # Build multi-level graph
                graph_data = self.graph_builder.build_graph(history)
                
                # GNN forward  
                gnn_output = self.gnn(
                    x=graph_data['node_features'],
                    edge_index=graph_data['edge_index'],
                    edge_type=graph_data.get('edge_type'),
                    node_type=graph_data.get('node_type')
                )
                
                # Aggregate: use mean of all node features
                graph_feat = gnn_output['node_features'].mean(dim=0, keepdim=True)  # [1, hidden_dim]
            
            graph_features_list.append(graph_feat)
        
        # Stack into batch
        graph_features = torch.cat(graph_features_list, dim=0)  # [batch, hidden_dim]
        
        return graph_features
    
    def forward(self,
                utterances: List[str],
                dialogue_histories: List[List[Dict]],
                previous_belief_states: List[Dict[str, str]],
                current_belief_states: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for batch training
        
        Args:
            utterances: List of current user utterances
            dialogue_histories: List of structured dialogue histories
            previous_belief_states: List of previous belief states
            current_belief_states: List of current belief states (for graph builder)
        
        Returns:
            gate_logits: [batch, num_slots, num_gate_classes]
            value_logits: [batch, num_slots, vocab_size]
        """
        batch_size = len(utterances)
        
        # 1. Encode utterances with BERT
        utterance_features = self.encode_utterance(utterances)  # [batch, hidden_dim]
        
        # 2. Process dialogue histories with graph
        graph_features = self.process_graph_batch(dialogue_histories)  # [batch, hidden_dim]
        
        # 3. Fuse utterance and graph features
        # Use multihead attention: query=utterance, key=graph, value=graph
        utterance_expanded = utterance_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        graph_expanded = graph_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        fused_features, _ = self.fusion(
            query=utterance_expanded,
            key=graph_expanded,
            value=graph_expanded
        )  # [batch, 1, hidden_dim]
        
        fused_features = fused_features.squeeze(1)  # [batch, hidden_dim]
        fused_features = self.dropout(fused_features)
        
        # 4. Predict for each slot
        gate_logits_list = []
        value_logits_list = []
        
        for slot_idx in range(self.num_slots):
            # Slot-specific projection
            slot_features = self.slot_projections[slot_idx](fused_features)  # [batch, hidden_dim]
            slot_features = torch.relu(slot_features)
            
            # Gate prediction
            gate_logits = self.gate_pred(slot_features)  # [batch, num_gate_classes]
            gate_logits_list.append(gate_logits)
            
            # Value prediction
            value_logits = self.value_pred(slot_features)  # [batch, vocab_size]
            value_logits_list.append(value_logits)
        
        # Stack into tensors
        gate_logits = torch.stack(gate_logits_list, dim=1)  # [batch, num_slots, num_gate_classes]
        value_logits = torch.stack(value_logits_list, dim=1)  # [batch, num_slots, vocab_size]
        
        return gate_logits, value_logits


if __name__ == "__main__":
    # Test model
    print("Testing BatchGraphDST...")
    
    import json
    
    # Load config
    slot_meta = json.load(open('data/processed_graph/slot_meta.json'))
    ontology = json.load(open('data/raw/ontology.json'))
    
    model = BatchGraphDST(
        slot_meta=slot_meta,
        value_vocab_size=1000,
        ontology=ontology,
        hidden_dim=768,
        gnn_layers=2
    )
    
    # Test batch
    utterances = ["I want a hotel in the north", "Book it for 2 people"]
    dialogue_histories = [[], [{'turn_id': 0, 'user': 'test', 'system': 'test', 'belief_state': {}, 'domains': [], 'timestamp': 0}]]
    previous_beliefs = [{}, {}]
    current_beliefs = [{'hotel-area': 'north'}, {'hotel-area': 'north', 'hotel-book people': '2'}]
    
    gate_logits, value_logits = model(utterances, dialogue_histories, previous_beliefs, current_beliefs)
    
    print(f"Gate logits shape: {gate_logits.shape}")  # [2, num_slots, 4]
    print(f"Value logits shape: {value_logits.shape}")  # [2, num_slots, vocab_size]
    print("✓ Model test passed!")
