"""
Multi-level Graph Builder

Xây dựng đồ thị đa tầng gồm:
1. Domain Graph: Domain nodes và relationships
2. Schema Graph: Slot-Value static relationships  
3. Value Graph: Dynamic value nodes từ dialogue history
4. Heterogeneous connections giữa các levels

Author: Assistant
Date: 2025-11-13
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


class MultiLevelGraphBuilder(nn.Module):
    """
    Multi-level Graph Builder cho History-Aware GraphDST
    
    Architecture:
    - Domain Level: 5 domain nodes [hotel, restaurant, attraction, train, taxi]
    - Schema Level: 30 slot nodes + value nodes từ ontology
    - Value Level: Dynamic value nodes từ dialogue history
    - Heterogeneous edges kết nối các levels
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 num_domains: int = 5,
                 max_history_turns: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.max_history_turns = max_history_turns
        self.dropout = dropout
        
        # Node type definitions
        self.NODE_TYPES = {
            'domain': 0,
            'slot': 1, 
            'value': 2,
            'turn': 3
        }
        
        # Edge type definitions
        self.EDGE_TYPES = {
            'domain_slot': 0,      # domain → slot
            'slot_value': 1,       # slot → value
            'turn_value': 2,       # turn → active values
            'temporal': 3,         # turn → turn (sequential)
            'cooccurrence': 4      # value ↔ value (co-occur in same turn)
        }
        
        # Initialize graph components
        self.domains = []
        self.slots = []  
        self.ontology_values = []
        self.slot_to_domain = {}
        self.slot_to_values = {}
        
        # Embeddings cho different node types
        self.domain_embeddings = nn.Embedding(num_domains, hidden_dim)
        self.slot_embeddings = nn.Embedding(50, hidden_dim)  # Max 50 slots
        self.value_embeddings = nn.Embedding(2000, hidden_dim)  # Max 2000 values
        self.turn_embeddings = nn.Embedding(max_history_turns, hidden_dim)
        
        # Position embeddings cho temporal modeling
        self.temporal_embeddings = nn.Embedding(max_history_turns, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embedding weights"""
        for embedding in [self.domain_embeddings, self.slot_embeddings, 
                         self.value_embeddings, self.turn_embeddings,
                         self.temporal_embeddings]:
            nn.init.xavier_uniform_(embedding.weight)
    
    def _ensure_domain_capacity(self, required_domains: int):
        """Resize domain embeddings if ontology introduces more domains"""
        if required_domains <= self.domain_embeddings.num_embeddings:
            return
        device = self.domain_embeddings.weight.device
        new_embedding = nn.Embedding(required_domains, self.hidden_dim).to(device)
        nn.init.xavier_uniform_(new_embedding.weight)
        with torch.no_grad():
            copy_count = self.domain_embeddings.num_embeddings
            new_embedding.weight[:copy_count] = self.domain_embeddings.weight[:copy_count]
        self.domain_embeddings = new_embedding
        self.num_domains = required_domains
    
    def load_ontology(self, slot_meta_path: str, ontology_path: str):
        """
        Load ontology data từ slot metadata và ontology files
        
        Args:
            slot_meta_path: Path to slot_meta.json
            ontology_path: Path to ontology.json
        """
        print("Loading multi-level ontology...")
        
        # Load slot metadata
        with open(slot_meta_path, 'r') as f:
            slot_data = json.load(f)
            if isinstance(slot_data, dict) and 'slot_meta' in slot_data:
                self.slots = slot_data['slot_meta']
            else:
                self.slots = slot_data
        
        # Load ontology for values
        with open(ontology_path, 'r') as f:
            ontology = json.load(f)
        
        # Extract domains từ slots
        self.domains = []
        for slot in self.slots:
            if '-' in slot:
                domain = slot.split('-')[0]
                if domain not in self.domains:
                    self.domains.append(domain)
                self.slot_to_domain[slot] = domain
        self._ensure_domain_capacity(len(self.domains))
        
        # Build slot-value mappings
        all_values = set()
        for slot, values in ontology.items():
            if slot in self.slots:
                if isinstance(values, list):
                    self.slot_to_values[slot] = values
                    all_values.update(values)
        
        self.ontology_values = sorted(list(all_values))
        
        # Create mappings
        self.domain_to_id = {domain: i for i, domain in enumerate(self.domains)}
        self.slot_to_id = {slot: i for i, slot in enumerate(self.slots)}
        self.value_to_id = {value: i for i, value in enumerate(self.ontology_values)}
        
        print(f"Loaded multi-level ontology:")
        print(f"  - Domains: {len(self.domains)} {self.domains}")
        print(f"  - Slots: {len(self.slots)}")
        print(f"  - Ontology Values: {len(self.ontology_values)}")
        
    def build_domain_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build Domain Graph
        
        Returns:
            domain_nodes: [num_domains, hidden_dim]
            domain_edges: [2, num_domain_edges] 
        """
        num_domains = len(self.domains)
        
        # Domain node features
        device = next(self.parameters()).device
        domain_ids = torch.arange(num_domains, device=device)
        domain_nodes = self.domain_embeddings(domain_ids)  # [num_domains, hidden_dim]
        
        # Domain-Domain edges (fully connected for cross-domain reasoning)
        edges = []
        for i in range(num_domains):
            for j in range(num_domains):
                if i != j:
                    edges.append([i, j])
        
        domain_edges = torch.tensor(edges, device=device).T if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        
        return domain_nodes, domain_edges
    
    def build_schema_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Schema Graph (Static)
        
        Returns:
            schema_nodes: [num_slots + num_values, hidden_dim]
            schema_edges: [2, num_schema_edges]
            node_types: [num_slots + num_values] (0=slot, 1=value)
        """
        num_slots = len(self.slots)
        num_values = len(self.ontology_values)
        
        # Slot node features
        device = next(self.parameters()).device
        slot_ids = torch.arange(num_slots, device=device)
        slot_nodes = self.slot_embeddings(slot_ids)  # [num_slots, hidden_dim]
        
        # Value node features  
        value_ids = torch.arange(num_values, device=device)
        value_nodes = self.value_embeddings(value_ids)  # [num_values, hidden_dim]
        
        # Combine nodes
        schema_nodes = torch.cat([slot_nodes, value_nodes], dim=0)  # [num_slots + num_values, hidden_dim]
        
        # Node type labels
        node_types = torch.cat([
            torch.zeros(num_slots, dtype=torch.long, device=device),  # 0 = slot
            torch.ones(num_values, dtype=torch.long, device=device)   # 1 = value
        ])
        
        # Build edges
        edges = []
        
        # Slot-Value edges
        for slot, values in self.slot_to_values.items():
            if slot in self.slot_to_id:
                slot_idx = self.slot_to_id[slot]
                for value in values:
                    if value in self.value_to_id:
                        value_idx = self.value_to_id[value] + num_slots  # Offset by slot count
                        edges.append([slot_idx, value_idx])  # slot → value
                        edges.append([value_idx, slot_idx])  # value → slot (bidirectional)
        
        schema_edges = torch.tensor(edges, device=device).T if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        
        return schema_nodes, schema_edges, node_types
    
    def build_value_graph_from_history(self, 
                                     dialogue_history: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Value Graph từ dialogue history (Dynamic)
        
        Args:
            dialogue_history: List of turns với belief states
            
        Returns:
            turn_nodes: [num_turns, hidden_dim]
            value_nodes: [num_active_values, hidden_dim] 
            turn_value_edges: [2, num_edges]
        """
        max_turns = min(len(dialogue_history), self.max_history_turns)
        
        if max_turns == 0:
            # Empty history
            device = next(self.parameters()).device
            return (torch.empty((0, self.hidden_dim), device=device), 
                   torch.empty((0, self.hidden_dim), device=device),
                   torch.empty((2, 0), dtype=torch.long, device=device))
        
        # Get device first
        device = next(self.parameters()).device
        
        # Turn node features với temporal position
        turn_ids = torch.arange(max_turns, device=device)
        turn_nodes = self.turn_embeddings(turn_ids) + self.temporal_embeddings(turn_ids)
        
        # Extract active values từ history
        active_values = set()
        turn_to_values = defaultdict(list)
        
        for turn_idx, turn in enumerate(dialogue_history[-max_turns:]):
            if 'belief_state' in turn:
                for slot_value in turn['belief_state']:
                    if len(slot_value) >= 2:
                        slot_name, value = slot_value[0], slot_value[1]
                        if value and value.lower() not in ['none', '']:
                            active_values.add(value)
                            turn_to_values[turn_idx].append(value)
        
        active_values = sorted(list(active_values))
        
        # Value node features cho active values
        if active_values:
            # Map active values to ontology values or create new embeddings
            value_indices = []
            for value in active_values:
                if value in self.value_to_id:
                    value_indices.append(self.value_to_id[value])
                else:
                    # Use a special embedding cho unseen values
                    value_indices.append(0)  # Default to first value embedding
            
            value_indices = torch.tensor(value_indices, device=device)
            value_nodes = self.value_embeddings(value_indices)
        else:
            value_nodes = torch.empty((0, self.hidden_dim), device=device)
        
        # Build turn-value edges
        edges = []
        value_to_idx = {value: i for i, value in enumerate(active_values)}
        
        for turn_idx, values in turn_to_values.items():
            for value in values:
                if value in value_to_idx:
                    value_idx = value_to_idx[value]
                    edges.append([turn_idx, len(turn_nodes) + value_idx])  # turn → value
        
        # Add temporal edges (turn → turn)
        for i in range(max_turns - 1):
            edges.append([i, i + 1])  # sequential turns
        
        turn_value_edges = torch.tensor(edges, device=device).T if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        
        return turn_nodes, value_nodes, turn_value_edges
    
    def build_unified_graph(self, 
                           dialogue_history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build unified heterogeneous graph combining all levels
        
        Args:
            dialogue_history: Current dialogue history
            
        Returns:
            Dictionary containing unified graph components
        """
        # Get device first for consistent usage
        device = next(self.parameters()).device
        
        # Build individual graphs
        domain_nodes, domain_edges = self.build_domain_graph()
        schema_nodes, schema_edges, schema_node_types = self.build_schema_graph()
        turn_nodes, history_value_nodes, turn_value_edges = self.build_value_graph_from_history(dialogue_history)
        
        # Combine all nodes
        all_nodes = [domain_nodes]
        node_offset = [0, len(domain_nodes)]
        
        if len(schema_nodes) > 0:
            all_nodes.append(schema_nodes)
            node_offset.append(node_offset[-1] + len(schema_nodes))
        
        if len(turn_nodes) > 0:
            all_nodes.append(turn_nodes)
            node_offset.append(node_offset[-1] + len(turn_nodes))
        
        if len(history_value_nodes) > 0:
            all_nodes.append(history_value_nodes)
            node_offset.append(node_offset[-1] + len(history_value_nodes))
        
        unified_nodes = torch.cat(all_nodes, dim=0)  # [total_nodes, hidden_dim]
        
        # Combine all edges với offset
        all_edges = []
        
        # Domain edges (no offset needed)
        if domain_edges.size(1) > 0:
            all_edges.append(domain_edges)
        
        # Schema edges (offset by domain count)
        if schema_edges.size(1) > 0:
            offset_schema_edges = schema_edges + node_offset[1]
            all_edges.append(offset_schema_edges)
        
        # Turn-value edges (offset by domain + schema count)
        if turn_value_edges.size(1) > 0:
            offset_turn_edges = turn_value_edges + node_offset[2]
            all_edges.append(offset_turn_edges)
        
        # Cross-level edges (Domain-Schema connections)
        cross_edges = []
        for slot in self.slots:
            if slot in self.slot_to_domain and slot in self.slot_to_id:
                domain = self.slot_to_domain[slot]
                if domain in self.domain_to_id:
                    domain_idx = self.domain_to_id[domain]
                    slot_idx = self.slot_to_id[slot] + node_offset[1]
                    cross_edges.append([domain_idx, slot_idx])  # domain → slot
                    cross_edges.append([slot_idx, domain_idx])  # slot → domain
        
        if cross_edges:
            cross_edge_tensor = torch.tensor(cross_edges, device=device).T
            all_edges.append(cross_edge_tensor)
        
        # Unified edge index (device already defined above)
        unified_edges = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Node type labels
        num_domains = len(domain_nodes)
        num_schema = len(schema_nodes) if len(schema_nodes) > 0 else 0
        num_turns = len(turn_nodes) if len(turn_nodes) > 0 else 0
        num_history_values = len(history_value_nodes) if len(history_value_nodes) > 0 else 0
        
        node_types = torch.cat([
            torch.full((num_domains,), self.NODE_TYPES['domain'], dtype=torch.long, device=device),
            torch.full((num_schema,), self.NODE_TYPES['slot'], dtype=torch.long, device=device) if num_schema > 0 else torch.empty(0, dtype=torch.long, device=device),
            torch.full((num_turns,), self.NODE_TYPES['turn'], dtype=torch.long, device=device) if num_turns > 0 else torch.empty(0, dtype=torch.long, device=device),
            torch.full((num_history_values,), self.NODE_TYPES['value'], dtype=torch.long, device=device) if num_history_values > 0 else torch.empty(0, dtype=torch.long, device=device)
        ])
        
        return {
            'node_features': unified_nodes,          # [total_nodes, hidden_dim]
            'edge_index': unified_edges,             # [2, total_edges] 
            'node_types': node_types,                # [total_nodes]
            'num_nodes': len(unified_nodes),
            'offsets': {
                'domain': (0, num_domains),
                'schema': (node_offset[1], node_offset[1] + num_schema) if num_schema > 0 else (0, 0),
                'turn': (node_offset[2], node_offset[2] + num_turns) if num_turns > 0 else (0, 0),
                'value': (node_offset[3], node_offset[3] + num_history_values) if num_history_values > 0 else (0, 0)
            }
        }
    
    def forward(self, dialogue_history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass để build complete multi-level graph
        
        Args:
            dialogue_history: Current dialogue history
            
        Returns:
            Multi-level graph structure
        """
        # Build unified heterogeneous graph
        graph_data = self.build_unified_graph(dialogue_history)
        
        # Apply dropout
        graph_data['node_features'] = self.dropout_layer(graph_data['node_features'])
        
        # Ensure all tensors are on the same device as the model
        device = next(self.parameters()).device
        for key in graph_data:
            if isinstance(graph_data[key], torch.Tensor):
                graph_data[key] = graph_data[key].to(device)
        
        return graph_data