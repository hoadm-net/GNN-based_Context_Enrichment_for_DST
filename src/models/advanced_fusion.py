"""
Advanced Fusion Layer

Multi-head attention fusion cho intent và graph context features.
Thay thế simple weighted sum bằng sophisticated cross-modal reasoning.

Author: Assistant  
Date: 2025-11-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math


class MultiModalAttentionFusion(nn.Module):
    """
    Multi-modal attention fusion cho BERT intent và GNN context
    
    Features:
    - Cross-attention between intent và graph features
    - Multi-head attention cho rich interactions
    - Gated fusion mechanism
    - Residual connections
    """
    
    def __init__(self,
                 intent_dim: int = 768,
                 graph_dim: int = 768,
                 fusion_dim: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        
        self.intent_dim = intent_dim
        self.graph_dim = graph_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gate = use_gate
        
        # Input projections để ensure consistent dimensions
        self.intent_proj = nn.Linear(intent_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        
        # Multi-head cross-attention: intent queries graph
        self.intent_to_graph_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head cross-attention: graph queries intent  
        self.graph_to_intent_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention cho refined features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated fusion mechanism
        if use_gate:
            self.fusion_gate = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),  # intent + graph + combined
                nn.Sigmoid()
            )
            
            self.intent_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),  # intent + graph_attended
                nn.Sigmoid()
            )
            
            self.graph_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),  # graph + intent_attended
                nn.Sigmoid()
            )
        
        # Output projections
        self.intent_output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.final_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.intent_norm1 = nn.LayerNorm(fusion_dim)
        self.intent_norm2 = nn.LayerNorm(fusion_dim)
        self.graph_norm1 = nn.LayerNorm(fusion_dim)
        self.graph_norm2 = nn.LayerNorm(fusion_dim)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self,
                intent_features: torch.Tensor,
                graph_features: torch.Tensor,
                intent_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Multi-modal attention fusion
        
        Args:
            intent_features: [batch, seq_len, intent_dim] từ BERT
            graph_features: [batch, 1, graph_dim] từ GNN 
            intent_mask: [batch, seq_len] attention mask cho intent
            return_attention: Return attention weights
            
        Returns:
            Dictionary containing fused features và attention weights
        """
        batch_size = intent_features.size(0)
        
        # Project inputs to common dimension
        intent_proj = self.intent_proj(intent_features)  # [batch, seq_len, fusion_dim]
        graph_proj = self.graph_proj(graph_features)     # [batch, 1, fusion_dim]
        
        # === Cross-Attention Phase ===
        
        # Intent queries Graph (Intent-to-Graph Attention)
        intent_attended_by_graph, i2g_attention = self.intent_to_graph_attention(
            query=intent_proj,         # [batch, seq_len, fusion_dim]
            key=graph_proj,           # [batch, 1, fusion_dim]  
            value=graph_proj,         # [batch, 1, fusion_dim]
            key_padding_mask=None     # Graph không có padding
        )  # [batch, seq_len, fusion_dim]
        
        # Graph queries Intent (Graph-to-Intent Attention)
        # Convert intent_mask to proper type
        intent_padding_mask = None
        if intent_mask is not None:
            intent_padding_mask = intent_mask.bool()
        
        graph_attended_by_intent, g2i_attention = self.graph_to_intent_attention(
            query=graph_proj,         # [batch, 1, fusion_dim]
            key=intent_proj,          # [batch, seq_len, fusion_dim]
            value=intent_proj,        # [batch, seq_len, fusion_dim]
            key_padding_mask=intent_padding_mask  # Mask padded tokens
        )  # [batch, 1, fusion_dim]
        
        # === Gated Fusion ===
        
        if self.use_gate:
            # Intent gate: balance original intent với graph-attended intent
            intent_gate_input = torch.cat([intent_proj, intent_attended_by_graph], dim=-1)
            intent_gate_weights = self.intent_gate(intent_gate_input)  # [batch, seq_len, fusion_dim]
            
            # Graph gate: balance original graph với intent-attended graph  
            graph_gate_input = torch.cat([graph_proj, graph_attended_by_intent], dim=-1)
            graph_gate_weights = self.graph_gate(graph_gate_input)  # [batch, 1, fusion_dim]
            
            # Apply gates
            gated_intent = intent_gate_weights * intent_proj + (1 - intent_gate_weights) * intent_attended_by_graph
            gated_graph = graph_gate_weights * graph_proj + (1 - graph_gate_weights) * graph_attended_by_intent
        else:
            # Simple residual connections
            gated_intent = intent_proj + intent_attended_by_graph
            gated_graph = graph_proj + graph_attended_by_intent
        
        # Layer normalization
        gated_intent = self.intent_norm1(gated_intent)
        gated_graph = self.graph_norm1(gated_graph)
        
        # === Self-Attention Phase ===
        
        # Combine intent và graph cho self-attention
        combined_features = torch.cat([gated_intent, gated_graph], dim=1)  # [batch, seq_len+1, fusion_dim]
        
        # Create combined mask
        if intent_mask is not None:
            intent_mask_bool = intent_mask.bool()
            graph_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=intent_mask.device)
            combined_mask = torch.cat([intent_mask_bool, graph_mask], dim=1)  # [batch, seq_len+1]
        else:
            combined_mask = None
        
        # Self-attention over combined features
        refined_features, self_attention = self.self_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features,
            key_padding_mask=combined_mask
        )  # [batch, seq_len+1, fusion_dim]
        
        # Split back intent và graph features
        refined_intent = refined_features[:, :-1, :]   # [batch, seq_len, fusion_dim]
        refined_graph = refined_features[:, -1:, :]    # [batch, 1, fusion_dim]
        
        # Residual connections + layer norm
        refined_intent = self.intent_norm2(gated_intent + refined_intent)
        refined_graph = self.graph_norm2(gated_graph + refined_graph)
        
        # === Output Projections ===
        
        # Enhanced intent features
        intent_with_context = torch.cat([gated_intent, refined_intent], dim=-1)
        enhanced_intent = self.intent_output_proj(intent_with_context)  # [batch, seq_len, fusion_dim]
        
        # Enhanced graph features  
        graph_with_context = torch.cat([gated_graph, refined_graph], dim=-1)
        enhanced_graph = self.graph_output_proj(graph_with_context)     # [batch, 1, fusion_dim]
        
        # Final fused representation
        # Global pooling cho intent features
        if intent_mask is not None:
            # Masked mean pooling
            intent_mask_bool = intent_mask.bool()
            intent_lengths = (~intent_mask_bool).sum(dim=1, keepdim=True).float()  # [batch, 1]
            intent_mask_expanded = intent_mask_bool.unsqueeze(-1).expand_as(enhanced_intent)
            masked_intent = enhanced_intent.masked_fill(intent_mask_expanded, 0.0)
            global_intent = masked_intent.sum(dim=1, keepdim=True) / intent_lengths.unsqueeze(-1)  # [batch, 1, fusion_dim]
        else:
            global_intent = torch.mean(enhanced_intent, dim=1, keepdim=True)  # [batch, 1, fusion_dim]
        
        # Combine global intent + enhanced graph
        final_combined = torch.cat([global_intent, enhanced_graph], dim=-1)  # [batch, 1, fusion_dim*2]
        fused_features = self.final_proj(final_combined)  # [batch, 1, fusion_dim]
        
        # === Prepare Output ===
        
        output = {
            'fused_features': fused_features,           # [batch, 1, fusion_dim]
            'enhanced_intent': enhanced_intent,         # [batch, seq_len, fusion_dim]
            'enhanced_graph': enhanced_graph,           # [batch, 1, fusion_dim]
            'global_intent': global_intent,             # [batch, 1, fusion_dim]
            'gated_intent': gated_intent,               # [batch, seq_len, fusion_dim]
            'gated_graph': gated_graph                  # [batch, 1, fusion_dim]
        }
        
        if return_attention:
            output['attention_weights'] = {
                'intent_to_graph': i2g_attention,       # [batch, num_heads, seq_len, 1]
                'graph_to_intent': g2i_attention,       # [batch, num_heads, 1, seq_len]  
                'self_attention': self_attention         # [batch, num_heads, seq_len+1, seq_len+1]
            }
        
        return output


class AdaptiveFusionLayer(nn.Module):
    """
    Adaptive Fusion Layer với learnable fusion weights
    
    Alternative cho fixed weighted sum approach
    """
    
    def __init__(self,
                 intent_dim: int = 768,
                 graph_dim: int = 768,
                 fusion_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        
        self.intent_dim = intent_dim
        self.graph_dim = graph_dim
        self.fusion_dim = fusion_dim
        
        # Adaptive weight networks
        self.intent_weight_net = nn.Sequential(
            nn.Linear(intent_dim + graph_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.graph_weight_net = nn.Sequential(
            nn.Linear(intent_dim + graph_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature projections
        self.intent_proj = nn.Linear(intent_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self,
                intent_features: torch.Tensor,
                graph_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adaptive fusion với learnable weights
        
        Args:
            intent_features: [batch, seq_len, intent_dim]
            graph_features: [batch, 1, graph_dim]
            
        Returns:
            Fused features dictionary
        """
        # Global pooling cho intent
        global_intent = torch.mean(intent_features, dim=1, keepdim=True)  # [batch, 1, intent_dim]
        
        # Concatenate cho weight computation
        combined_input = torch.cat([global_intent, graph_features], dim=-1)  # [batch, 1, intent_dim + graph_dim]
        
        # Compute adaptive weights
        intent_weight = self.intent_weight_net(combined_input)  # [batch, 1, 1]
        graph_weight = self.graph_weight_net(combined_input)   # [batch, 1, 1]
        
        # Normalize weights
        total_weight = intent_weight + graph_weight
        intent_weight = intent_weight / (total_weight + 1e-8)
        graph_weight = graph_weight / (total_weight + 1e-8)
        
        # Project features
        projected_intent = self.intent_proj(global_intent)    # [batch, 1, fusion_dim]
        projected_graph = self.graph_proj(graph_features)     # [batch, 1, fusion_dim]
        
        # Weighted fusion
        fused = intent_weight * projected_intent + graph_weight * projected_graph
        
        # Output projection
        fused_features = self.output_proj(fused)  # [batch, 1, fusion_dim]
        
        return {
            'fused_features': fused_features,
            'intent_weight': intent_weight,
            'graph_weight': graph_weight,
            'projected_intent': projected_intent,
            'projected_graph': projected_graph
        }


# Test functions
def test_multimodal_fusion():
    """Test MultiModalAttentionFusion"""
    print("Testing Multi-modal Attention Fusion...")
    
    batch_size = 2
    seq_len = 20
    intent_dim = 768
    graph_dim = 768
    
    # Sample data
    intent_features = torch.randn(batch_size, seq_len, intent_dim)
    graph_features = torch.randn(batch_size, 1, graph_dim)
    intent_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    intent_mask[:, -5:] = True  # Mask last 5 tokens
    
    # Initialize fusion layer
    fusion = MultiModalAttentionFusion(
        intent_dim=intent_dim,
        graph_dim=graph_dim,
        fusion_dim=768,
        num_heads=8
    )
    
    # Forward pass
    with torch.no_grad():
        output = fusion(intent_features, graph_features, intent_mask, return_attention=True)
    
    print(f"Fused features shape: {output['fused_features'].shape}")
    print(f"Enhanced intent shape: {output['enhanced_intent'].shape}")
    print(f"Enhanced graph shape: {output['enhanced_graph'].shape}")
    print("✅ Multi-modal fusion test passed!")


def test_adaptive_fusion():
    """Test AdaptiveFusionLayer"""
    print("Testing Adaptive Fusion Layer...")
    
    batch_size = 2
    seq_len = 20
    intent_dim = 768
    graph_dim = 768
    
    # Sample data
    intent_features = torch.randn(batch_size, seq_len, intent_dim)
    graph_features = torch.randn(batch_size, 1, graph_dim)
    
    # Initialize adaptive fusion
    fusion = AdaptiveFusionLayer(
        intent_dim=intent_dim,
        graph_dim=graph_dim,
        fusion_dim=768
    )
    
    # Forward pass
    with torch.no_grad():
        output = fusion(intent_features, graph_features)
    
    print(f"Fused features shape: {output['fused_features'].shape}")
    print(f"Intent weight: {output['intent_weight'].mean().item():.3f}")
    print(f"Graph weight: {output['graph_weight'].mean().item():.3f}")
    print("✅ Adaptive fusion test passed!")


if __name__ == "__main__":
    test_multimodal_fusion()
    test_adaptive_fusion()