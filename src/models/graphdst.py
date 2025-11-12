"""
GraphDST Model adapted for current MultiWOZ 2.4 data pipeline

This module implements the Graph-Enhanced Dialog State Tracking model
based on the original GraphDST architecture but adapted to work with
the current data preprocessing pipeline.

Key adaptations:
- 30 slots instead of 37 (based on current slot_meta.json)
- 5 domains: hotel, restaurant, attraction, train, taxi
- Integrated with current data loading pipeline
- Compatible with current evaluation framework
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import numpy as np


@dataclass
class GraphDSTConfig:
    """Configuration for GraphDST model adapted to current data"""
    # Model dimensions
    hidden_dim: int = 768
    num_gnn_layers: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Text encoder
    text_encoder: str = "bert-base-uncased"
    max_sequence_length: int = 512
    
    # Graph structure (adapted to current data)
    num_domains: int = 5  # hotel, restaurant, attraction, train, taxi
    num_slots: int = 30   # Current slot_meta.json has 30 slots
    max_turn_length: int = 20
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Value prediction
    max_categorical_values: int = 100  # For categorical slot values


class MultiHeadGraphAttention(nn.Module):
    """Multi-head attention for graph neural networks"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert output_dim % num_heads == 0
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.output_dim
        )
        
        return self.out_linear(attn_output)


class SchemaGraphLayer(nn.Module):
    """Schema-aware Graph Convolution Layer for DST"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Message functions for different edge types
        self.domain_to_slot = nn.Linear(input_dim, output_dim)
        self.slot_to_domain = nn.Linear(input_dim, output_dim)
        self.slot_to_slot = nn.Linear(input_dim, output_dim)
        
        # Update functions
        self.domain_update = nn.Linear(input_dim + output_dim, output_dim)
        self.slot_update = nn.Linear(input_dim + output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, domain_features: torch.Tensor, slot_features: torch.Tensor,
                domain_slot_edges: torch.Tensor, slot_slot_edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            domain_features: (num_domains, input_dim)
            slot_features: (num_slots, input_dim)  
            domain_slot_edges: (2, num_domain_slot_edges) - edges between domains and slots
            slot_slot_edges: (2, num_slot_slot_edges) - edges between slots
        """
        
        # Domain to slot messages
        domain_messages = torch.zeros_like(slot_features)
        if domain_slot_edges.size(1) > 0:
            src_domains = domain_slot_edges[0]  # domain indices
            tgt_slots = domain_slot_edges[1]    # slot indices
            
            # Get domain features and transform
            src_features = domain_features[src_domains]  # (num_edges, input_dim)
            messages = self.domain_to_slot(src_features)  # (num_edges, output_dim)
            
            # Aggregate messages to slots
            domain_messages.index_add_(0, tgt_slots, messages)
        
        # Slot to domain messages
        slot_messages = torch.zeros_like(domain_features)
        if domain_slot_edges.size(1) > 0:
            src_slots = domain_slot_edges[1]    # slot indices  
            tgt_domains = domain_slot_edges[0]  # domain indices
            
            src_features = slot_features[src_slots]
            messages = self.slot_to_domain(src_features)
            
            slot_messages.index_add_(0, tgt_domains, messages)
        
        # Slot to slot messages
        slot_to_slot_messages = torch.zeros_like(slot_features)
        if slot_slot_edges.size(1) > 0:
            src_slots = slot_slot_edges[0]
            tgt_slots = slot_slot_edges[1]
            
            src_features = slot_features[src_slots]
            messages = self.slot_to_slot(src_features)
            
            slot_to_slot_messages.index_add_(0, tgt_slots, messages)
        
        # Update domain features
        domain_input = torch.cat([domain_features, slot_messages], dim=-1)
        new_domain_features = self.domain_update(domain_input)
        new_domain_features = F.relu(new_domain_features)
        new_domain_features = self.dropout(new_domain_features)
        new_domain_features = self.layer_norm(new_domain_features)
        
        # Update slot features
        total_slot_messages = domain_messages + slot_to_slot_messages
        slot_input = torch.cat([slot_features, total_slot_messages], dim=-1)
        new_slot_features = self.slot_update(slot_input)
        new_slot_features = F.relu(new_slot_features)
        new_slot_features = self.dropout(new_slot_features)
        new_slot_features = self.layer_norm(new_slot_features)
        
        return new_domain_features, new_slot_features


class CrossDomainAttentionLayer(nn.Module):
    """Cross-domain attention for knowledge sharing"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.domain_attention = MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
        self.slot_attention = MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
        
        self.domain_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.slot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, domain_features: torch.Tensor, slot_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add batch dimension if needed
        if domain_features.dim() == 2:
            domain_features = domain_features.unsqueeze(0)  # (1, num_domains, hidden_dim)
        if slot_features.dim() == 2:
            slot_features = slot_features.unsqueeze(0)  # (1, num_slots, hidden_dim)
        
        # Domain self-attention
        domain_attn_out = self.domain_attention(domain_features, domain_features, domain_features)
        domain_features = self.layer_norm1(domain_features + domain_attn_out)
        domain_ffn_out = self.domain_ffn(domain_features)
        domain_features = self.layer_norm2(domain_features + domain_ffn_out)
        
        # Slot self-attention
        slot_attn_out = self.slot_attention(slot_features, slot_features, slot_features)
        slot_features = self.layer_norm1(slot_features + slot_attn_out)
        slot_ffn_out = self.slot_ffn(slot_features)
        slot_features = self.layer_norm2(slot_features + slot_ffn_out)
        
        # Remove batch dimension
        domain_features = domain_features.squeeze(0)  # (num_domains, hidden_dim)
        slot_features = slot_features.squeeze(0)      # (num_slots, hidden_dim)
        
        return domain_features, slot_features


class TemporalDialogEncoder(nn.Module):
    """Temporal encoding for dialog history"""
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional output back to hidden_dim
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Turn-level attention
        self.turn_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, dialog_sequence: torch.Tensor, 
                turn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            dialog_sequence: (batch, max_turns, hidden_dim)
            turn_mask: (batch, max_turns) - mask for valid turns
        """
        batch_size, max_turns, _ = dialog_sequence.size()
        
        # GRU encoding
        gru_output, _ = self.gru(dialog_sequence)  # (batch, max_turns, hidden_dim * 2)
        gru_output = self.output_proj(gru_output)  # (batch, max_turns, hidden_dim)
        
        # Self-attention over turns
        if turn_mask is not None:
            # Convert mask for attention (True = valid, False = padding)
            attn_mask = ~turn_mask.bool()  # Invert for attention mask
        else:
            attn_mask = None
        
        attn_output, _ = self.turn_attention(
            query=gru_output,
            key=gru_output, 
            value=gru_output,
            key_padding_mask=attn_mask
        )
        
        # Residual connection
        output = gru_output + attn_output
        
        return output


class MultiTaskDSTHeads(nn.Module):
    """Multi-task prediction heads for DST"""
    
    def __init__(self, config: GraphDSTConfig, slot_names: List[str]):
        super().__init__()
        self.config = config
        self.slot_names = slot_names
        self.num_slots = len(slot_names)
        
        hidden_dim = config.hidden_dim
        
        # Domain classification head (multi-label)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, config.num_domains)
        )
        
        # Slot activation heads (binary classification for each slot)
        self.slot_classifiers = nn.ModuleDict()
        for slot_name in slot_names:
            slot_key = slot_name.replace('-', '_').replace(' ', '_')
            self.slot_classifiers[slot_key] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # concat slot + utterance features
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, 2)  # Binary: active or not
            )
        
        # Value prediction heads (simplified - assume all are span-based for now)
        self.value_start_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.value_end_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, utterance_features: torch.Tensor, 
                slot_features: torch.Tensor,
                domain_features: torch.Tensor,
                sequence_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            utterance_features: (batch, hidden_dim) - CLS token features
            slot_features: (num_slots, hidden_dim) - slot node features
            domain_features: (num_domains, hidden_dim) - domain node features  
            sequence_features: (batch, seq_len, hidden_dim) - for span prediction
        """
        batch_size = utterance_features.size(0)
        predictions = {}
        
        # 1. Domain prediction (multi-label classification)
        domain_logits = self.domain_classifier(utterance_features)
        predictions['domains'] = torch.sigmoid(domain_logits)  # (batch, num_domains)
        
        # 2. Slot activation prediction
        slot_predictions = {}
        for i, slot_name in enumerate(self.slot_names):
            slot_key = slot_name.replace('-', '_').replace(' ', '_')
            
            # Get slot-specific features
            slot_feature = slot_features[i:i+1]  # (1, hidden_dim)
            slot_feature_expanded = slot_feature.expand(batch_size, -1)  # (batch, hidden_dim)
            
            # Combine utterance and slot features
            combined_features = torch.cat([utterance_features, slot_feature_expanded], dim=-1)
            
            # Predict slot activation
            slot_logits = self.slot_classifiers[slot_key](combined_features)
            slot_predictions[slot_name] = slot_logits  # (batch, 2)
        
        predictions['slot_activations'] = slot_predictions
        
        # 3. Value span prediction (if sequence features available)
        if sequence_features is not None:
            seq_len = sequence_features.size(1)
            
            # Expand utterance features to match sequence length
            utterance_expanded = utterance_features.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine sequence and utterance features
            combined_seq_features = torch.cat([sequence_features, utterance_expanded], dim=-1)
            
            # Predict start and end positions
            start_logits = self.value_start_classifier(combined_seq_features).squeeze(-1)  # (batch, seq_len)
            end_logits = self.value_end_classifier(combined_seq_features).squeeze(-1)    # (batch, seq_len)
            
            predictions['span_start'] = start_logits
            predictions['span_end'] = end_logits
        
        return predictions


class GraphDSTModel(nn.Module):
    """Main GraphDST Model adapted for current data pipeline"""
    
    def __init__(self, config: GraphDSTConfig, slot_names: List[str]):
        super().__init__()
        self.config = config
        self.slot_names = slot_names
        self.num_slots = len(slot_names)
        
        # Text encoder (BERT)
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        
        # Feature projection to match hidden_dim
        encoder_dim = self.text_encoder.config.hidden_size
        if encoder_dim != config.hidden_dim:
            self.feature_projection = nn.Linear(encoder_dim, config.hidden_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # Initialize learnable node embeddings
        self.domain_embeddings = nn.Parameter(torch.randn(config.num_domains, config.hidden_dim))
        self.slot_embeddings = nn.Parameter(torch.randn(self.num_slots, config.hidden_dim))
        
        # Graph neural network layers
        self.schema_layers = nn.ModuleList([
            SchemaGraphLayer(config.hidden_dim, config.hidden_dim, config.dropout)
            for _ in range(config.num_gnn_layers)
        ])
        
        self.cross_attention_layers = nn.ModuleList([
            CrossDomainAttentionLayer(config.hidden_dim, config.num_attention_heads, config.dropout)
            for _ in range(config.num_gnn_layers)
        ])
        
        # Temporal dialog encoder
        self.dialog_encoder = TemporalDialogEncoder(config.hidden_dim, num_layers=2, dropout=config.dropout)
        
        # Multi-task prediction heads
        self.prediction_heads = MultiTaskDSTHeads(config, slot_names)
        
        # Initialize domain-slot connections (based on slot names)
        self.register_buffer('domain_slot_edges', self._build_domain_slot_edges())
        self.register_buffer('slot_slot_edges', self._build_slot_slot_edges())
        
        # Initialize parameters
        self._init_parameters()
    
    def _build_domain_slot_edges(self) -> torch.Tensor:
        """Build edges between domains and slots based on slot names"""
        domain_names = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
        
        edges = []
        for slot_idx, slot_name in enumerate(self.slot_names):
            # Find which domain this slot belongs to
            for domain_idx, domain_name in enumerate(domain_names):
                if slot_name.startswith(domain_name):
                    edges.append([domain_idx, slot_idx])
                    break
        
        if not edges:
            # Fallback: create some default connections
            edges = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        
        return torch.tensor(edges, dtype=torch.long).t()  # (2, num_edges)
    
    def _build_slot_slot_edges(self) -> torch.Tensor:
        """Build edges between related slots (same domain)"""
        edges = []
        
        # Group slots by domain
        domain_slots = {}
        domain_names = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
        
        for slot_idx, slot_name in enumerate(self.slot_names):
            for domain_name in domain_names:
                if slot_name.startswith(domain_name):
                    if domain_name not in domain_slots:
                        domain_slots[domain_name] = []
                    domain_slots[domain_name].append(slot_idx)
                    break
        
        # Create edges within same domain
        for domain_name, slot_indices in domain_slots.items():
            for i, slot_i in enumerate(slot_indices):
                for j, slot_j in enumerate(slot_indices):
                    if i != j:  # Connect all slots within same domain
                        edges.append([slot_i, slot_j])
        
        if not edges:
            # Fallback: create some default connections
            for i in range(min(5, self.num_slots)):
                for j in range(min(5, self.num_slots)):
                    if i != j:
                        edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.domain_embeddings, std=0.02)
        nn.init.normal_(self.slot_embeddings, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GraphDST model
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) - optional
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Text encoding
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get features
        sequence_features = encoder_outputs.last_hidden_state  # (batch, seq_len, encoder_dim)
        cls_features = sequence_features[:, 0]  # (batch, encoder_dim)
        
        # Project to hidden dimension
        sequence_features = self.feature_projection(sequence_features)  # (batch, seq_len, hidden_dim)
        cls_features = self.feature_projection(cls_features)  # (batch, hidden_dim)
        
        # 2. Schema graph processing
        domain_features = self.domain_embeddings  # (num_domains, hidden_dim)
        slot_features = self.slot_embeddings      # (num_slots, hidden_dim)
        
        # Apply GNN layers
        for schema_layer, cross_attn_layer in zip(self.schema_layers, self.cross_attention_layers):
            # Schema-aware graph convolution
            domain_features, slot_features = schema_layer(
                domain_features, slot_features,
                self.domain_slot_edges, self.slot_slot_edges
            )
            
            # Cross-domain attention
            domain_features, slot_features = cross_attn_layer(domain_features, slot_features)
        
        # 3. Multi-task prediction
        predictions = self.prediction_heads(
            utterance_features=cls_features,
            slot_features=slot_features,
            domain_features=domain_features,
            sequence_features=sequence_features
        )
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        
        # Domain loss (multi-label BCE)
        if 'domain_labels' in labels:
            domain_loss = F.binary_cross_entropy(
                predictions['domains'],
                labels['domain_labels'].float()
            )
            losses['domain'] = domain_loss
        
        # Slot activation losses
        slot_losses = []
        for slot_name in self.slot_names:
            label_key = f"{slot_name}_active"
            if label_key in labels and slot_name in predictions['slot_activations']:
                slot_loss = F.cross_entropy(
                    predictions['slot_activations'][slot_name],
                    labels[label_key].long()
                )
                slot_losses.append(slot_loss)
        
        if slot_losses:
            losses['slot'] = torch.stack(slot_losses).mean()
        
        # Span prediction losses
        if 'span_start' in predictions and 'span_start_labels' in labels:
            start_loss = F.cross_entropy(
                predictions['span_start'],
                labels['span_start_labels'].long(),
                ignore_index=-1
            )
            losses['span_start'] = start_loss
        
        if 'span_end' in predictions and 'span_end_labels' in labels:
            end_loss = F.cross_entropy(
                predictions['span_end'],
                labels['span_end_labels'].long(),
                ignore_index=-1
            )
            losses['span_end'] = end_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


def create_graphdst_model(slot_meta_path: str, config: Optional[GraphDSTConfig] = None) -> GraphDSTModel:
    """Factory function to create GraphDST model"""
    
    # Load slot metadata
    with open(slot_meta_path, 'r') as f:
        slot_data = json.load(f)
        slot_names = slot_data['slot_meta']
    
    # Use default config if not provided
    if config is None:
        config = GraphDSTConfig(num_slots=len(slot_names))
    else:
        config.num_slots = len(slot_names)
    
    # Create model
    model = GraphDSTModel(config, slot_names)
    
    print(f"Created GraphDST model with:")
    print(f"  - {len(slot_names)} slots")
    print(f"  - {config.num_domains} domains") 
    print(f"  - {config.hidden_dim} hidden dimensions")
    print(f"  - {config.num_gnn_layers} GNN layers")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("GraphDST Model for Current Data Pipeline")
    print("=" * 60)
    
    # Load current slot metadata
    slot_meta_path = "data/processed/slot_meta.json"
    
    config = GraphDSTConfig(
        hidden_dim=768,
        num_gnn_layers=3,
        num_attention_heads=8,
        dropout=0.1
    )
    
    model = create_graphdst_model(slot_meta_path, config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    
    print("Predictions:")
    for key, value in predictions.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} items")
            for subkey, subvalue in list(value.items())[:3]:  # Show first 3
                print(f"    {subkey}: {subvalue.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ GraphDST model created successfully!")