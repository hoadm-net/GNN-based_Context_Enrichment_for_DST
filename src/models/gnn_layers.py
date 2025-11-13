"""
GNN Layers - Schema-aware Graph Neural Networks

Implements specialized GNN layers for History-Aware GraphDST:
1. Schema-aware GCN: Graph convolution với schema knowledge
2. Cross-domain GAT: Graph attention across different domains
3. Temporal reasoning GNN: Handle temporal relationships trong history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class SchemaAwareGCN(nn.Module):
    """
    Schema-aware Graph Convolutional Network
    
    Key features:
    - Type-aware message passing: Different weights cho different node types
    - Schema-guided aggregation: Use ontology structure để guide information flow
    - Residual connections: Preserve original features
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_node_types: int = 6,  # Turn, BeliefState, SlotValue, Domain, Slot, Value
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        
        # Type-specific transformation matrices
        self.node_type_transforms = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_node_types)
        ])
        
        # Message passing weights
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregation weights
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm và residual
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Edge type embedding cho different edge types
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(32, hidden_dim // 4),  # Assuming edge_attr_dim = 32
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim]
            node_types: Node type indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        
        num_nodes = x.size(0)
        
        # Apply type-specific transformations
        if node_types is not None:
            h = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            for node_type in range(self.num_node_types):
                mask = (node_types == node_type)
                if mask.any():
                    h[mask] = self.node_type_transforms[node_type](x[mask])
        else:
            h = self.node_type_transforms[0](x)  # Use first transform as default
        
        # Message passing
        if edge_index.size(1) > 0:
            row, col = edge_index
            
            # Compute messages
            messages = self.message_mlp(torch.cat([h[row], h[col]], dim=1))
            
            # Apply edge weights nếu có edge attributes
            if edge_attr is not None:
                edge_weights = torch.sigmoid(self.edge_type_mlp(edge_attr))
                messages = messages * edge_weights
            
            # Aggregate messages
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, col, messages)
            
            # Apply aggregation MLP
            aggregated = self.aggregate_mlp(aggregated)
            
            # Update node features
            h = h + aggregated
        
        # Output projection
        h = self.output_proj(h)
        
        # Residual connection và layer norm
        if h.size(-1) == x.size(-1):
            h = self.layer_norm(h + x)
        else:
            h = self.layer_norm(h)
        
        return self.dropout(h)


class CrossDomainGAT(nn.Module):
    """
    Cross-domain Graph Attention Network
    
    Key features:
    - Multi-head attention: Multiple attention heads cho diverse relationships
    - Cross-domain focus: Specially designed để connect different domains
    - Semantic attention: Use semantic similarity để guide attention
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 alpha: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.alpha = alpha
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear transformations cho Q, K, V
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Attention mechanism
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm và dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        
        num_nodes = x.size(0)
        
        # Linear transformations
        Q = self.W_q(x).view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        K = self.W_k(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize output
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        
        if edge_index.size(1) > 0:
            row, col = edge_index
            
            # Compute attention scores
            # [num_edges, num_heads]
            attention_scores = (Q[row] * K[col]).sum(dim=-1) / math.sqrt(self.head_dim)
            attention_scores = self.leaky_relu(attention_scores)
            
            # Apply edge attributes nếu có
            if edge_attr is not None:
                # Simple linear transformation của edge attributes
                edge_weights = torch.sigmoid(edge_attr.sum(dim=1, keepdim=True))  # [num_edges, 1]
                attention_scores = attention_scores * edge_weights
            
            # Softmax over neighbors cho mỗi node
            attention_probs = torch.zeros_like(attention_scores)
            for node in range(num_nodes):
                neighbor_mask = (col == node)
                if neighbor_mask.any():
                    neighbor_scores = attention_scores[neighbor_mask]
                    neighbor_probs = self.softmax(neighbor_scores)
                    attention_probs[neighbor_mask] = neighbor_probs
            
            # Apply attention to values
            # [num_edges, num_heads, head_dim]
            weighted_values = attention_probs.unsqueeze(-1) * V[row]
            
            # Aggregate by target nodes
            for i in range(len(col)):
                out[col[i]] += weighted_values[i]
        
        # Reshape output
        out = out.view(num_nodes, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection và layer norm
        if out.size(-1) == x.size(-1):
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)
        
        return self.dropout(out)


class TemporalReasoningGNN(nn.Module):
    """
    Temporal Reasoning Graph Neural Network
    
    Key features:
    - Temporal encoding: Encode temporal relationships between dialog turns
    - Sequential processing: Handle sequential nature của dialog
    - Memory mechanism: Maintain temporal context across turns
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 max_sequence_length: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Temporal position encoding
        self.temporal_encoding = nn.Embedding(max_sequence_length, hidden_dim)
        
        # LSTM cho sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if hidden_dim > 1 else 0.0
        )
        
        # Attention mechanism cho temporal context
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Graph convolution cho temporal edges
        self.temporal_gcn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layers
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                temporal_positions: Optional[torch.Tensor] = None,
                turn_boundaries: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            temporal_positions: Temporal position cho mỗi node [num_nodes]
            turn_boundaries: List of (start_idx, end_idx) cho mỗi turn
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        
        num_nodes = x.size(0)
        
        # Add temporal encoding
        if temporal_positions is not None:
            # Clamp positions to valid range
            temporal_positions = torch.clamp(temporal_positions, 0, self.max_sequence_length - 1)
            temporal_embed = self.temporal_encoding(temporal_positions)
            
            # Project input nếu cần
            if x.size(-1) != self.hidden_dim:
                x = nn.Linear(x.size(-1), self.hidden_dim, device=x.device)(x)
            
            h = x + temporal_embed
        else:
            h = x
        
        # Sequential processing nếu có turn boundaries
        if turn_boundaries is not None and len(turn_boundaries) > 0:
            # Process each turn sequence
            turn_outputs = []
            for start_idx, end_idx in turn_boundaries:
                if start_idx < end_idx and end_idx <= num_nodes:
                    turn_features = h[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, hidden_dim]
                    
                    # LSTM processing
                    lstm_out, _ = self.lstm(turn_features)
                    
                    # Self-attention within turn
                    attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
                    
                    turn_outputs.append(attn_out.squeeze(0))
                else:
                    # Fallback cho invalid boundaries
                    turn_outputs.append(h[start_idx:end_idx] if start_idx < num_nodes else h[start_idx:start_idx+1])
            
            # Concatenate turn outputs
            if turn_outputs:
                h = torch.cat(turn_outputs, dim=0)
        
        # Temporal graph convolution
        if edge_index.size(1) > 0:
            row, col = edge_index
            
            # Create messages
            messages = self.temporal_gcn(torch.cat([h[row], h[col]], dim=1))
            
            # Aggregate messages
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, col, messages)
            
            # Update features
            h = h + aggregated
        
        # Output projection
        h = self.output_proj(h)
        h = self.layer_norm(h)
        
        return self.dropout(h)


class UnifiedGNNLayer(nn.Module):
    """
    Unified GNN Layer combining all GNN components
    
    Combines:
    - Schema-aware GCN
    - Cross-domain GAT  
    - Temporal reasoning GNN
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_node_types: int = 6,
                 num_heads: int = 8,
                 max_sequence_length: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Three GNN components
        self.schema_gcn = SchemaAwareGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_node_types=num_node_types,
            dropout=dropout
        )
        
        self.cross_gat = CrossDomainGAT(
            input_dim=hidden_dim,  # Takes output từ schema_gcn
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.temporal_gnn = TemporalReasoningGNN(
            input_dim=hidden_dim,  # Takes output từ cross_gat
            hidden_dim=hidden_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                node_types: Optional[torch.Tensor] = None,
                temporal_positions: Optional[torch.Tensor] = None,
                turn_boundaries: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Forward pass through all GNN components
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]  
            edge_attr: Edge attributes [num_edges, edge_attr_dim]
            node_types: Node type indices [num_nodes]
            temporal_positions: Temporal positions [num_nodes]
            turn_boundaries: Turn boundary information
            
        Returns:
            Final node representations [num_nodes, hidden_dim]
        """
        
        # Schema-aware processing
        h1 = self.schema_gcn(x, edge_index, edge_attr, node_types)
        
        # Cross-domain attention
        h2 = self.cross_gat(h1, edge_index, edge_attr)
        
        # Temporal reasoning
        h3 = self.temporal_gnn(h2, edge_index, temporal_positions, turn_boundaries)
        
        # Fusion của all representations
        h_fused = self.fusion_layer(torch.cat([h1, h2, h3], dim=-1))
        
        return self.layer_norm(h_fused)


# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing GNN Layers...")
    
    # Test parameters
    num_nodes = 20
    input_dim = 768
    hidden_dim = 768
    num_edges = 30
    edge_attr_dim = 32
    
    # Create test data
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    node_types = torch.randint(0, 6, (num_nodes,))
    temporal_positions = torch.arange(num_nodes) % 10
    turn_boundaries = [(0, 5), (5, 10), (10, 15), (15, 20)]
    
    # Test SchemaAwareGCN
    print("Testing SchemaAwareGCN...")
    schema_gcn = SchemaAwareGCN(input_dim=input_dim, hidden_dim=hidden_dim)
    h1 = schema_gcn(x, edge_index, edge_attr, node_types)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {h1.shape}")
    print(f"  Output range: [{h1.min():.3f}, {h1.max():.3f}]")
    
    # Test CrossDomainGAT
    print("\nTesting CrossDomainGAT...")
    cross_gat = CrossDomainGAT(input_dim=hidden_dim, hidden_dim=hidden_dim)
    h2 = cross_gat(h1, edge_index, edge_attr)
    print(f"  Input shape: {h1.shape}")
    print(f"  Output shape: {h2.shape}")
    print(f"  Output range: [{h2.min():.3f}, {h2.max():.3f}]")
    
    # Test TemporalReasoningGNN
    print("\nTesting TemporalReasoningGNN...")
    temporal_gnn = TemporalReasoningGNN(input_dim=hidden_dim, hidden_dim=hidden_dim)
    h3 = temporal_gnn(h2, edge_index, temporal_positions, turn_boundaries)
    print(f"  Input shape: {h2.shape}")
    print(f"  Output shape: {h3.shape}")
    print(f"  Output range: [{h3.min():.3f}, {h3.max():.3f}]")
    
    # Test UnifiedGNNLayer
    print("\nTesting UnifiedGNNLayer...")
    unified_gnn = UnifiedGNNLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_node_types=6,
        num_heads=8,
        max_sequence_length=50
    )
    
    h_final = unified_gnn(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_types=node_types,
        temporal_positions=temporal_positions,
        turn_boundaries=turn_boundaries
    )
    
    print(f"  Input shape: {x.shape}")
    print(f"  Final output shape: {h_final.shape}")
    print(f"  Final output range: [{h_final.min():.3f}, {h_final.max():.3f}]")
    
    # Test parameter count
    total_params = sum(p.numel() for p in unified_gnn.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("✅ GNN Layers testing completed!")