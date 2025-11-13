"""
Schema Graph Builder - Static Ontology Structure

Xây dựng static graph structure từ MultiWOZ ontology.
Graph này represent domain-slot-value relationships và không thay đổi theo dialog.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from collections import defaultdict


class SchemaNodeType(Enum):
    """Types of nodes trong schema graph"""
    DOMAIN = "domain"
    SLOT = "slot" 
    VALUE = "value"


class SchemaEdgeType(Enum):
    """Types of edges trong schema graph"""
    DOMAIN_SLOT = "domain_slot"     # Domain → Slot (contains)
    SLOT_VALUE = "slot_value"       # Slot → Value (accepts)
    SLOT_SLOT = "slot_slot"         # Slot ↔ Slot (co-occurrence)
    VALUE_VALUE = "value_value"     # Value ↔ Value (similarity)


@dataclass
class SchemaGraph:
    """Complete schema graph structure"""
    node_features: torch.Tensor      # [num_nodes, hidden_dim]
    edge_index: torch.Tensor         # [2, num_edges]
    edge_attr: torch.Tensor          # [num_edges, edge_attr_dim]
    node_types: List[SchemaNodeType]
    node_ids: List[str]
    
    # Type-specific indices
    domain_indices: List[int]
    slot_indices: List[int]
    value_indices: List[int]
    
    # Mappings
    id_to_idx: Dict[str, int]
    idx_to_id: Dict[int, str]
    
    metadata: Dict[str, Any]


class SchemaGraphBuilder(nn.Module):
    """
    Constructs static schema graph từ ontology
    
    Key features:
    - Static structure: Không thay đổi theo dialog
    - Rich relationships: Domain-slot-value connections
    - Pre-trained embeddings: Based on semantic similarity
    - Co-occurrence patterns: Learned từ training data
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 edge_attr_dim: int = 32,
                 use_pretrained_embeddings: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        
        # Node embeddings (sẽ được initialized từ ontology)
        self.domain_embeddings = None
        self.slot_embeddings = None  
        self.value_embeddings = None
        
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(len(SchemaEdgeType), edge_attr_dim)
        
        # Feature projection layers
        self.domain_projector = nn.Linear(hidden_dim, hidden_dim)
        self.slot_projector = nn.Linear(hidden_dim, hidden_dim)
        self.value_projector = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Ontology data
        self.domains = []
        self.slots = []
        self.values = []
        self.slot_to_domain = {}
        self.slot_to_values = {}
        
    def load_ontology(self, slot_meta_path: str, vocab_path: str):
        """
        Load ontology data từ slot metadata và vocabularies
        
        Args:
            slot_meta_path: Path to slot metadata JSON
            vocab_path: Path to slot vocabularies JSON
        """
        
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_meta = json.load(f)
            
        # Load vocabularies
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Extract domains, slots, values
        self.domains = list(set([
            slot.split('-')[0] for slot in slot_meta.keys()
            if '-' in slot
        ]))
        
        self.slots = list(slot_meta.keys())
        
        # Build slot to domain mapping
        for slot in self.slots:
            if '-' in slot:
                domain = slot.split('-')[0]
                self.slot_to_domain[slot] = domain
        
        # Extract all values và build slot-value mapping 
        all_values = set()
        self.slot_to_values = {}
        
        for slot, values in vocab_data.items():
            if slot in self.slots:
                self.slot_to_values[slot] = values
                all_values.update(values)
        
        self.values = sorted(list(all_values))
        
        print(f"Loaded ontology:")
        print(f"  - Domains: {len(self.domains)}")
        print(f"  - Slots: {len(self.slots)}")
        print(f"  - Values: {len(self.values)}")
        
        # Initialize embeddings
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize node embeddings"""
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(len(self.domains), self.hidden_dim)
        
        # Slot embeddings
        self.slot_embeddings = nn.Embedding(len(self.slots), self.hidden_dim)
        
        # Value embeddings
        self.value_embeddings = nn.Embedding(len(self.values), self.hidden_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.domain_embeddings.weight)
        nn.init.xavier_uniform_(self.slot_embeddings.weight)
        nn.init.xavier_uniform_(self.value_embeddings.weight)
        
    def _create_domain_slot_edges(self) -> List[Tuple[str, str, SchemaEdgeType]]:
        """Create edges giữa domains và slots"""
        
        edges = []
        for slot, domain in self.slot_to_domain.items():
            edges.append((f"domain_{domain}", f"slot_{slot}", SchemaEdgeType.DOMAIN_SLOT))
            
        return edges
    
    def _create_slot_value_edges(self) -> List[Tuple[str, str, SchemaEdgeType]]:
        """Create edges giữa slots và values"""
        
        edges = []
        for slot, values in self.slot_to_values.items():
            for value in values:
                edges.append((f"slot_{slot}", f"value_{value}", SchemaEdgeType.SLOT_VALUE))
                
        return edges
    
    def _create_slot_slot_edges(self, 
                               co_occurrence_threshold: float = 0.1) -> List[Tuple[str, str, SchemaEdgeType]]:
        """
        Create edges giữa slots based on co-occurrence patterns
        
        Args:
            co_occurrence_threshold: Minimum co-occurrence để create edge
        """
        
        edges = []
        
        # Simple heuristic: slots trong cùng domain có higher co-occurrence
        domain_slots = defaultdict(list)
        for slot, domain in self.slot_to_domain.items():
            domain_slots[domain].append(slot)
        
        # Connect slots within same domain
        for domain, slots in domain_slots.items():
            for i, slot1 in enumerate(slots):
                for slot2 in slots[i+1:]:
                    edges.append((f"slot_{slot1}", f"slot_{slot2}", SchemaEdgeType.SLOT_SLOT))
                    edges.append((f"slot_{slot2}", f"slot_{slot1}", SchemaEdgeType.SLOT_SLOT))
        
        # Cross-domain connections cho related concepts
        cross_domain_pairs = [
            ('hotel-area', 'restaurant-area'),
            ('hotel-pricerange', 'restaurant-pricerange'),
            ('train-departure', 'taxi-departure'), 
            ('train-destination', 'taxi-destination')
        ]
        
        for slot1, slot2 in cross_domain_pairs:
            if slot1 in self.slots and slot2 in self.slots:
                edges.append((f"slot_{slot1}", f"slot_{slot2}", SchemaEdgeType.SLOT_SLOT))
                edges.append((f"slot_{slot2}", f"slot_{slot1}", SchemaEdgeType.SLOT_SLOT))
        
        return edges
    
    def build_graph(self) -> SchemaGraph:
        """
        Build complete schema graph
        
        Returns:
            SchemaGraph object
        """
        
        # Create node lists và mappings
        node_ids = []
        node_types = []
        
        # Add domain nodes
        domain_indices = []
        for i, domain in enumerate(self.domains):
            node_id = f"domain_{domain}"
            node_ids.append(node_id)
            node_types.append(SchemaNodeType.DOMAIN)
            domain_indices.append(len(node_ids) - 1)
        
        # Add slot nodes
        slot_indices = []
        for i, slot in enumerate(self.slots):
            node_id = f"slot_{slot}"
            node_ids.append(node_id)
            node_types.append(SchemaNodeType.SLOT)
            slot_indices.append(len(node_ids) - 1)
        
        # Add value nodes
        value_indices = []
        for i, value in enumerate(self.values):
            node_id = f"value_{value}"
            node_ids.append(node_id)
            node_types.append(SchemaNodeType.VALUE)
            value_indices.append(len(node_ids) - 1)
        
        # Create mappings
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        idx_to_id = {idx: node_id for idx, node_id in enumerate(node_ids)}
        
        # Create node features
        node_features = []
        
        # Domain features
        for i, domain in enumerate(self.domains):
            domain_emb = self.domain_embeddings(torch.tensor(i))
            domain_feat = self.layer_norm(self.domain_projector(domain_emb))
            node_features.append(domain_feat)
        
        # Slot features  
        for i, slot in enumerate(self.slots):
            slot_emb = self.slot_embeddings(torch.tensor(i))
            slot_feat = self.layer_norm(self.slot_projector(slot_emb))
            node_features.append(slot_feat)
        
        # Value features
        for i, value in enumerate(self.values):
            value_emb = self.value_embeddings(torch.tensor(i))
            value_feat = self.layer_norm(self.value_projector(value_emb))
            node_features.append(value_feat)
        
        node_features = torch.stack(node_features)
        
        # Create edges
        all_edges = []
        all_edges.extend(self._create_domain_slot_edges())
        all_edges.extend(self._create_slot_value_edges())
        all_edges.extend(self._create_slot_slot_edges())
        
        # Convert edges to tensor format
        edge_indices = []
        edge_attributes = []
        
        for source_id, target_id, edge_type in all_edges:
            if source_id in id_to_idx and target_id in id_to_idx:
                source_idx = id_to_idx[source_id]
                target_idx = id_to_idx[target_id]
                
                edge_indices.append([source_idx, target_idx])
                
                # Edge attribute
                edge_type_id = list(SchemaEdgeType).index(edge_type)
                edge_attr = self.edge_type_embedding(torch.tensor(edge_type_id))
                edge_attributes.append(edge_attr)
        
        edge_index = torch.tensor(edge_indices).t().contiguous() if edge_indices else torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.stack(edge_attributes) if edge_attributes else torch.empty(0, self.edge_attr_dim)
        
        return SchemaGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            node_ids=node_ids,
            domain_indices=domain_indices,
            slot_indices=slot_indices,
            value_indices=value_indices,
            id_to_idx=id_to_idx,
            idx_to_id=idx_to_id,
            metadata={
                'num_nodes': len(node_ids),
                'num_edges': edge_index.size(1),
                'num_domains': len(self.domains),
                'num_slots': len(self.slots),
                'num_values': len(self.values),
                'domains': self.domains,
                'slots': self.slots,
                'values': self.values
            }
        )
    
    def get_node_embeddings_by_type(self, 
                                   schema_graph: SchemaGraph,
                                   node_type: SchemaNodeType) -> torch.Tensor:
        """
        Extract node embeddings cho specific node type
        
        Args:
            schema_graph: Schema graph object
            node_type: Type of nodes to extract
            
        Returns:
            Node embeddings tensor
        """
        
        if node_type == SchemaNodeType.DOMAIN:
            indices = schema_graph.domain_indices
        elif node_type == SchemaNodeType.SLOT:
            indices = schema_graph.slot_indices
        elif node_type == SchemaNodeType.VALUE:
            indices = schema_graph.value_indices
        else:
            raise ValueError(f"Unknown node type: {node_type}")
        
        return schema_graph.node_features[indices]


# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing Schema Graph Builder...")
    
    # Create builder
    builder = SchemaGraphBuilder(hidden_dim=768)
    
    # Mock ontology data
    import tempfile
    import os
    
    mock_slot_meta = {
        'hotel-area': {'description': 'Area of hotel'},
        'hotel-pricerange': {'description': 'Price range of hotel'},
        'restaurant-food': {'description': 'Type of food'},
        'restaurant-pricerange': {'description': 'Price range of restaurant'},
        'train-departure': {'description': 'Train departure location'}
    }
    
    mock_vocab = {
        'hotel-area': ['center', 'north', 'south', 'east', 'west'],
        'hotel-pricerange': ['cheap', 'moderate', 'expensive'],
        'restaurant-food': ['chinese', 'italian', 'indian', 'french'],
        'restaurant-pricerange': ['cheap', 'moderate', 'expensive'],
        'train-departure': ['cambridge', 'london', 'birmingham']
    }
    
    # Save mock data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_slot_meta, f)
        slot_meta_path = f.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_vocab, f)
        vocab_path = f.name
    
    # Load ontology
    builder.load_ontology(slot_meta_path, vocab_path)
    
    # Build schema graph
    with torch.no_grad():
        schema_graph = builder.build_graph()
    
    print(f"Built schema graph:")
    print(f"  - Total nodes: {schema_graph.metadata['num_nodes']}")
    print(f"  - Total edges: {schema_graph.metadata['num_edges']}")
    print(f"  - Domains: {schema_graph.metadata['num_domains']}")
    print(f"  - Slots: {schema_graph.metadata['num_slots']}")
    print(f"  - Values: {schema_graph.metadata['num_values']}")
    print(f"  - Node features shape: {schema_graph.node_features.shape}")
    print(f"  - Edge index shape: {schema_graph.edge_index.shape}")
    print(f"  - Edge attributes shape: {schema_graph.edge_attr.shape}")
    
    # Test node type filtering
    domain_embeddings = builder.get_node_embeddings_by_type(schema_graph, SchemaNodeType.DOMAIN)
    slot_embeddings = builder.get_node_embeddings_by_type(schema_graph, SchemaNodeType.SLOT)
    value_embeddings = builder.get_node_embeddings_by_type(schema_graph, SchemaNodeType.VALUE)
    
    print(f"  - Domain embeddings shape: {domain_embeddings.shape}")
    print(f"  - Slot embeddings shape: {slot_embeddings.shape}")
    print(f"  - Value embeddings shape: {value_embeddings.shape}")
    
    # Clean up
    os.unlink(slot_meta_path)
    os.unlink(vocab_path)
    
    print("✅ Schema Graph Builder testing completed!")