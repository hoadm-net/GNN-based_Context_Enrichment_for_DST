"""
History Graph Builder - Dynamic Graph Construction từ Dialog History

Xây dựng dynamic graph structure từ previous belief states và dialog history.
Mỗi dialog sẽ có graph structure riêng dựa trên lịch sử conversation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import json


class NodeType(Enum):
    """Types of nodes trong history graph"""
    TURN = "turn"
    BELIEF_STATE = "belief_state" 
    SLOT_VALUE = "slot_value"


class EdgeType(Enum):
    """Types of edges trong history graph"""
    TEMPORAL = "temporal"           # Turn t-1 → Turn t
    GENERATES = "generates"         # Turn → Belief State
    CONTAINS = "contains"           # Belief State → Slot-Value
    EVOLVES = "evolves"            # BS t-1 → BS t (state evolution)
    DEPENDENCY = "dependency"       # Related slots


@dataclass
class GraphNode:
    """Represents a node trong history graph"""
    node_id: str
    node_type: NodeType
    features: torch.Tensor
    metadata: Dict[str, Any]
    
    
@dataclass
class GraphEdge:
    """Represents an edge trong history graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class HistoryGraph:
    """Complete history graph structure"""
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    node_features: torch.Tensor  # [num_nodes, hidden_dim]
    edge_index: torch.Tensor     # [2, num_edges]
    edge_attr: torch.Tensor      # [num_edges, edge_attr_dim]
    node_types: List[NodeType]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert HistoryGraph to dictionary format for model processing"""
        # Convert node types to integer indices
        node_type_indices = []
        for node_type in self.node_types:
            if node_type == NodeType.TURN:
                node_type_indices.append(0)
            elif node_type == NodeType.BELIEF_STATE:
                node_type_indices.append(1)
            elif node_type == NodeType.SLOT_VALUE:
                node_type_indices.append(2)
        
        # Extract turn boundaries from metadata
        turn_boundaries = self.metadata.get('turn_boundaries', [])
        
        return {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'node_types': torch.tensor(node_type_indices),
            'turn_boundaries': turn_boundaries
        }


class HistoryGraphBuilder(nn.Module):
    """
    Constructs dynamic graph from dialog history
    
    Key features:
    - Dynamic nodes: Created from actual dialog history
    - Temporal edges: Capture dialog evolution
    - Rich structure: Multiple node/edge types
    - Learnable embeddings: Node features updated during training
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 max_history_turns: int = 5,
                 edge_attr_dim: int = 32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_history_turns = max_history_turns
        self.edge_attr_dim = edge_attr_dim
        
        # Node embeddings cho different node types
        self.turn_embedding = nn.Embedding(100, hidden_dim)  # Max 100 turns
        self.slot_embedding = nn.Embedding(50, hidden_dim)   # Max 50 slots
        self.value_embedding = nn.Embedding(2000, hidden_dim) # Max 2000 values
        
        # Edge embeddings
        self.edge_type_embedding = nn.Embedding(len(EdgeType), edge_attr_dim)
        
        # Feature projection layers
        self.turn_projector = nn.Linear(hidden_dim, hidden_dim)
        self.belief_state_projector = nn.Linear(hidden_dim, hidden_dim)
        self.slot_value_projector = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Slot name to ID mapping (sẽ được load từ vocab)
        self.slot_to_id = {}
        self.value_to_id = {}
        
    def load_vocabularies(self, slot_vocab_path: str):
        """Load slot và value vocabularies"""
        with open(slot_vocab_path, 'r') as f:
            vocab_data = json.load(f)
            
        # Build slot mapping
        all_slots = []
        for slot, values in vocab_data.items():
            all_slots.append(slot)
            
        self.slot_to_id = {slot: idx for idx, slot in enumerate(all_slots)}
        
        # Build value mapping (global across all slots)
        all_values = set()
        for slot, values in vocab_data.items():
            all_values.update(values)
            
        self.value_to_id = {value: idx for idx, value in enumerate(sorted(all_values))}
        
    def create_turn_node(self, 
                        turn_id: int, 
                        turn_data: Dict) -> GraphNode:
        """Create node cho dialog turn"""
        
        # Turn embedding based on position
        turn_features = self.turn_embedding(torch.tensor(turn_id))
        turn_features = self.layer_norm(self.turn_projector(turn_features))
        
        node = GraphNode(
            node_id=f"turn_{turn_id}",
            node_type=NodeType.TURN,
            features=turn_features,
            metadata={
                'turn_id': turn_id,
                'utterance': turn_data.get('utterance', ''),
                'speaker': turn_data.get('speaker', 'user'),
                'timestamp': turn_id
            }
        )
        
        return node
        
    def create_belief_state_node(self, 
                                turn_id: int, 
                                belief_state: Dict) -> GraphNode:
        """Create node cho belief state snapshot"""
        
        # Aggregate features từ slot-value pairs
        slot_features = []
        for slot, value in belief_state.items():
            if slot in self.slot_to_id and value in self.value_to_id:
                slot_emb = self.slot_embedding(torch.tensor(self.slot_to_id[slot]))
                value_emb = self.value_embedding(torch.tensor(self.value_to_id[value]))
                slot_features.append(slot_emb + value_emb)
                
        if slot_features:
            # Mean pooling của slot features
            bs_features = torch.stack(slot_features).mean(dim=0)
        else:
            # Default features nếu empty belief state
            bs_features = torch.zeros(self.hidden_dim)
            
        bs_features = self.layer_norm(self.belief_state_projector(bs_features))
        
        node = GraphNode(
            node_id=f"bs_{turn_id}",
            node_type=NodeType.BELIEF_STATE,
            features=bs_features,
            metadata={
                'turn_id': turn_id,
                'belief_state': belief_state,
                'num_slots': len(belief_state)
            }
        )
        
        return node
        
    def create_slot_value_node(self, 
                              slot: str, 
                              value: str, 
                              turn_id: int) -> GraphNode:
        """Create node cho individual slot-value assignment"""
        
        if slot not in self.slot_to_id or value not in self.value_to_id:
            # Create default features for unknown slot/value
            sv_features = torch.randn(self.hidden_dim) * 0.1
        else:
            slot_emb = self.slot_embedding(torch.tensor(self.slot_to_id[slot]))
            value_emb = self.value_embedding(torch.tensor(self.value_to_id[value]))
            sv_features = slot_emb + value_emb
            
        sv_features = self.layer_norm(self.slot_value_projector(sv_features))
        
        node = GraphNode(
            node_id=f"sv_{slot}_{value}_{turn_id}",
            node_type=NodeType.SLOT_VALUE,
            features=sv_features,
            metadata={
                'slot': slot,
                'value': value,
                'turn_id': turn_id
            }
        )
        
        return node
    
    def build_graph(self, dialog_history: List[Dict]) -> HistoryGraph:
        """
        Build complete history graph từ dialog history
        
        Args:
            dialog_history: List of turns với belief states
            
        Returns:
            HistoryGraph object
        """
        
        nodes = {}
        edges = []
        
        # Limit history length
        recent_history = dialog_history[-self.max_history_turns:]
        
        previous_turn_id = None
        previous_bs_id = None
        
        # Create nodes for each turn
        for turn_idx, turn_data in enumerate(recent_history):
            turn_id = turn_data.get('turn_id', turn_idx)
            
            # 1. Create turn node
            turn_node = self.create_turn_node(turn_id, turn_data)
            nodes[turn_node.node_id] = turn_node
            
            # 2. Create temporal edge with previous turn
            if previous_turn_id is not None:
                temporal_edge = GraphEdge(
                    source_id=f"turn_{previous_turn_id}",
                    target_id=turn_node.node_id,
                    edge_type=EdgeType.TEMPORAL,
                    weight=1.0
                )
                edges.append(temporal_edge)
            
            # 3. Create belief state node if exists
            belief_state = turn_data.get('belief_state', {})
            if belief_state:
                bs_node = self.create_belief_state_node(turn_id, belief_state)
                nodes[bs_node.node_id] = bs_node
                
                # Edge: Turn → Belief State
                generates_edge = GraphEdge(
                    source_id=turn_node.node_id,
                    target_id=bs_node.node_id,
                    edge_type=EdgeType.GENERATES,
                    weight=1.0
                )
                edges.append(generates_edge)
                
                # Edge: Previous BS → Current BS (state evolution)
                if previous_bs_id is not None:
                    evolves_edge = GraphEdge(
                        source_id=previous_bs_id,
                        target_id=bs_node.node_id,
                        edge_type=EdgeType.EVOLVES,
                        weight=1.0
                    )
                    edges.append(evolves_edge)
                
                # 4. Create slot-value nodes
                for slot, value in belief_state.items():
                    sv_node = self.create_slot_value_node(slot, value, turn_id)
                    nodes[sv_node.node_id] = sv_node
                    
                    # Edge: Belief State → Slot-Value
                    contains_edge = GraphEdge(
                        source_id=bs_node.node_id,
                        target_id=sv_node.node_id,
                        edge_type=EdgeType.CONTAINS,
                        weight=1.0
                    )
                    edges.append(contains_edge)
                
                previous_bs_id = bs_node.node_id
            
            previous_turn_id = turn_id
        
        # Convert to tensor format
        return self._convert_to_tensors(nodes, edges)
    
    def _convert_to_tensors(self, 
                           nodes: Dict[str, GraphNode], 
                           edges: List[GraphEdge]) -> HistoryGraph:
        """Convert nodes và edges to tensor format"""
        
        if not nodes:
            # Empty graph
            return HistoryGraph(
                nodes=nodes,
                edges=edges,
                node_features=torch.empty(0, self.hidden_dim),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, self.edge_attr_dim),
                node_types=[],
                metadata={'num_nodes': 0, 'num_edges': 0}
            )
        
        # Create node ID to index mapping
        node_ids = list(nodes.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Stack node features
        node_features = torch.stack([nodes[node_id].features for node_id in node_ids])
        node_types = [nodes[node_id].node_type for node_id in node_ids]
        
        # Create edge index và edge attributes
        if edges:
            edge_indices = []
            edge_attributes = []
            
            for edge in edges:
                if edge.source_id in id_to_idx and edge.target_id in id_to_idx:
                    source_idx = id_to_idx[edge.source_id]
                    target_idx = id_to_idx[edge.target_id]
                    
                    edge_indices.append([source_idx, target_idx])
                    
                    # Edge attribute from edge type
                    edge_type_id = list(EdgeType).index(edge.edge_type)
                    edge_attr = self.edge_type_embedding(torch.tensor(edge_type_id))
                    edge_attributes.append(edge_attr)
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices).t().contiguous()
                edge_attr = torch.stack(edge_attributes)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, self.edge_attr_dim)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, self.edge_attr_dim)
        
        return HistoryGraph(
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            metadata={
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'node_ids': node_ids,
                'id_to_idx': id_to_idx
            }
        )


# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing History Graph Builder...")
    
    # Create builder
    builder = HistoryGraphBuilder(hidden_dim=768)
    
    # Mock slot vocabularies
    mock_vocab = {
        'hotel-area': ['center', 'north', 'south'],
        'hotel-pricerange': ['cheap', 'moderate', 'expensive'],
        'restaurant-food': ['chinese', 'italian', 'indian']
    }
    
    # Save mock vocab để test
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_vocab, f)
        vocab_path = f.name
    
    builder.load_vocabularies(vocab_path)
    
    # Mock dialog history
    dialog_history = [
        {
            'turn_id': 0,
            'utterance': 'I need a hotel',
            'speaker': 'user',
            'belief_state': {}
        },
        {
            'turn_id': 1,
            'utterance': 'I want something cheap',
            'speaker': 'user', 
            'belief_state': {'hotel-pricerange': 'cheap'}
        },
        {
            'turn_id': 2,
            'utterance': 'In the center please',
            'speaker': 'user',
            'belief_state': {'hotel-pricerange': 'cheap', 'hotel-area': 'center'}
        }
    ]
    
    print(f"Dialog history: {len(dialog_history)} turns")
    
    # Build graph
    with torch.no_grad():
        history_graph = builder.build_graph(dialog_history)
    
    print(f"Built graph:")
    print(f"  - Nodes: {history_graph.metadata['num_nodes']}")
    print(f"  - Edges: {history_graph.metadata['num_edges']}")
    print(f"  - Node features shape: {history_graph.node_features.shape}")
    print(f"  - Edge index shape: {history_graph.edge_index.shape}")
    print(f"  - Edge attributes shape: {history_graph.edge_attr.shape}")
    
    # Print node types
    node_type_counts = {}
    for node_type in history_graph.node_types:
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    print(f"  - Node type distribution: {node_type_counts}")
    
    # Clean up
    os.unlink(vocab_path)
    
    print("✅ History Graph Builder testing completed!")