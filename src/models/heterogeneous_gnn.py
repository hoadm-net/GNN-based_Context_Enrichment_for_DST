"""
Heterogeneous Graph Neural Network Layers

Specialized GNN layers cho different node types:
- Domain nodes: Cross-domain reasoning
- Slot nodes: Schema relationships  
- Value nodes: Value interactions
- Turn nodes: Temporal dialogue flow

Author: Assistant
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, List, Tuple, Optional, Any
import math


class HeterogeneousGNNLayer(MessagePassing):
    """
    Heterogeneous GNN Layer cho different node types
    
    Features:
    - Type-specific transformations
    - Cross-type message passing
    - Attention-based aggregation
    """
    
    def __init__(self,
                 in_dim: int = 768,
                 out_dim: int = 768,
                 num_node_types: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Type-specific transformations
        self.type_transforms = nn.ModuleDict({
            'domain': nn.Linear(in_dim, out_dim),
            'slot': nn.Linear(in_dim, out_dim),
            'value': nn.Linear(in_dim, out_dim),
            'turn': nn.Linear(in_dim, out_dim)
        })
        
        # Multi-head attention cho message passing
        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # Message transformation
        self.message_transform = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Type embeddings để distinguish node types
        self.type_embeddings = nn.Embedding(num_node_types, out_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters"""
        for transform in self.type_transforms.values():
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)
        
        nn.init.xavier_uniform_(self.type_embeddings.weight)
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                node_types: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            node_types: Node type IDs [num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Type-specific transformations
        h_transformed = torch.zeros(x.size(0), self.out_dim, device=x.device)
        
        # Apply type-specific transforms
        type_names = ['domain', 'slot', 'value', 'turn']
        for type_id, type_name in enumerate(type_names):
            mask = node_types == type_id
            if mask.any():
                h_transformed[mask] = self.type_transforms[type_name](x[mask])
        
        # Add type embeddings
        type_emb = self.type_embeddings(node_types)
        h_transformed = h_transformed + type_emb
        
        # Message passing
        if edge_index.size(1) > 0:
            out = self.propagate(edge_index, x=h_transformed)
        else:
            out = h_transformed
        
        # Combine original features với messages
        output = self.output_proj(torch.cat([h_transformed, out], dim=-1))
        
        return output
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Create messages từ source nodes (j) to target nodes (i)
        
        Args:
            x_i: Target node features [num_edges, out_dim]
            x_j: Source node features [num_edges, out_dim]
            
        Returns:
            Messages [num_edges, out_dim]
        """
        # Concatenate source và target features
        messages = torch.cat([x_i, x_j], dim=-1)  # [num_edges, 2*out_dim]
        
        # Transform messages
        messages = self.message_transform(messages)  # [num_edges, out_dim]
        
        return messages


class TemporalGRULayer(nn.Module):
    """
    GRU Layer cho temporal modeling của dialogue turns
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism cho important turns
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self,
                turn_features: torch.Tensor,
                turn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass cho temporal modeling
        
        Args:
            turn_features: [batch, num_turns, input_dim]
            turn_mask: [batch, num_turns] attention mask
            
        Returns:
            Temporal-enhanced features [batch, num_turns, hidden_dim]
        """
        batch_size, num_turns = turn_features.shape[:2]
        
        # GRU temporal modeling
        gru_output, _ = self.gru(turn_features)  # [batch, num_turns, hidden_dim * directions]
        
        # Self-attention cho important turns
        if turn_mask is not None:
            # Convert mask to attention mask
            attn_mask = ~turn_mask  # True values are masked out
        else:
            attn_mask = None
        
        attended_output, attention_weights = self.temporal_attention(
            query=gru_output,
            key=gru_output,
            value=gru_output,
            key_padding_mask=attn_mask
        )  # [batch, num_turns, hidden_dim * directions]
        
        # Residual connection
        output = gru_output + attended_output
        
        # Final projection
        output = self.output_proj(output)  # [batch, num_turns, hidden_dim]
        
        return output


class HeterogeneousGNN(nn.Module):
    """
    Complete Heterogeneous GNN Module
    
    Architecture:
    - Multiple HeterogeneousGNNLayers
    - Temporal GRU cho turn sequences
    - Cross-type reasoning
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_layers: int = 3,
                 num_node_types: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 temporal_layers: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_node_types = num_node_types
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Heterogeneous GNN layers
        self.gnn_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_node_types=num_node_types,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Temporal GRU cho turn modeling
        self.temporal_gru = TemporalGRULayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global graph pooling
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat graph + temporal
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self,
                graph_data: Dict[str, torch.Tensor],
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass cho Heterogeneous GNN
        
        Args:
            graph_data: Dictionary containing graph components
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing processed graph features
        """
        node_features = graph_data['node_features']  # [num_nodes, input_dim]
        edge_index = graph_data['edge_index']        # [2, num_edges]
        node_types = graph_data['node_types']        # [num_nodes]
        offsets = graph_data['offsets']
        
        # Input projection
        h = self.input_proj(node_features)  # [num_nodes, hidden_dim]
        
        # Multi-layer heterogeneous GNN
        attention_weights = {}
        
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index, node_types)
            
            # Residual connection + layer norm
            h = layer_norm(h + h_new)
        
        # Extract turn features cho temporal modeling
        turn_start, turn_end = offsets['turn']
        if turn_end > turn_start:
            turn_features = h[turn_start:turn_end]  # [num_turns, hidden_dim]
            
            # Add batch dimension for GRU
            turn_features = turn_features.unsqueeze(0)  # [1, num_turns, hidden_dim]
            
            # Apply temporal GRU
            temporal_features = self.temporal_gru(turn_features)  # [1, num_turns, hidden_dim]
            temporal_features = temporal_features.squeeze(0)     # [num_turns, hidden_dim]
            
            # Update turn features in graph without in-place ops on views
            h = h.clone()
            h[turn_start:turn_end] = temporal_features
        
        # Global graph representation
        graph_repr = torch.mean(h, dim=0, keepdim=True)  # [1, hidden_dim]
        graph_repr = self.graph_pooling(graph_repr)
        
        # Temporal representation (mean của turn features)
        if turn_end > turn_start:
            temporal_repr = torch.mean(h[turn_start:turn_end], dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            temporal_repr = torch.zeros_like(graph_repr)
        
        # Combine graph + temporal representations
        combined_repr = torch.cat([graph_repr, temporal_repr], dim=-1)  # [1, hidden_dim * 2]
        final_repr = self.output_proj(combined_repr)  # [1, hidden_dim]
        
        output = {
            'graph_features': final_repr,           # [1, hidden_dim]
            'node_features': h,                     # [num_nodes, hidden_dim]
            'graph_representation': graph_repr,     # [1, hidden_dim] 
            'temporal_representation': temporal_repr # [1, hidden_dim]
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output


# Test function
def test_heterogeneous_gnn():
    """Test Heterogeneous GNN implementation"""
    print("Testing Heterogeneous GNN...")
    
    # Create sample data
    num_nodes = 50
    input_dim = 768
    hidden_dim = 768
    
    # Sample graph data
    node_features = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    node_types = torch.randint(0, 4, (num_nodes,))
    
    graph_data = {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_types': node_types,
        'offsets': {
            'domain': (0, 5),
            'schema': (5, 35),
            'turn': (35, 45),
            'value': (45, 50)
        }
    }
    
    # Initialize model
    model = HeterogeneousGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_node_types=4
    )
    
    # Forward pass
    with torch.no_grad():
        output = model(graph_data, return_attention=True)
    
    print(f"Graph features shape: {output['graph_features'].shape}")
    print(f"Node features shape: {output['node_features'].shape}")
    print("✅ Heterogeneous GNN test passed!")


if __name__ == "__main__":
    test_heterogeneous_gnn()