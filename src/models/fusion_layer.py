"""
Fusion Mechanism - Cross-Modal Attention

Implements cross-attention mechanism to fuse:
1. Intent features from BERT (current utterance)
2. Context features from GNN (historical + schema information)

Key innovation: Separate processing branches merged through sophisticated attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism between intent và context features
    
    Design:
    - Intent features: BERT output cho current utterance [768 dim]
    - Context features: GNN output cho history + schema [768 dim]  
    - Cross-attention: Allow intent to attend to relevant context
    """
    
    def __init__(self,
                 intent_dim: int = 768,
                 context_dim: int = 768,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        
        self.intent_dim = intent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Project features to same dimension nếu cần
        self.intent_proj = nn.Linear(intent_dim, hidden_dim) if intent_dim != hidden_dim else nn.Identity()
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
        
        # Cross-attention layers: Intent attends to Context  
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # Query từ intent
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # Key từ context
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)  # Value từ context
        
        # Reverse attention: Context attends to Intent
        self.q_proj_rev = nn.Linear(hidden_dim, hidden_dim)  # Query từ context
        self.k_proj_rev = nn.Linear(hidden_dim, hidden_dim)  # Key từ intent
        self.v_proj_rev = nn.Linear(hidden_dim, hidden_dim)  # Value từ intent
        
        # Output projections
        self.intent_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norms
        self.intent_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                intent_features: torch.Tensor,
                context_features: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention forward pass
        
        Args:
            intent_features: Intent features từ BERT [batch_size, seq_len, intent_dim]
            context_features: Context features từ GNN [batch_size, num_nodes, context_dim]
            context_mask: Mask cho context nodes [batch_size, num_nodes]
            
        Returns:
            Tuple of (enhanced_intent_features, enhanced_context_features)
        """
        
        batch_size = intent_features.size(0)
        intent_seq_len = intent_features.size(1)
        context_seq_len = context_features.size(1)
        
        # Project features
        intent_h = self.intent_proj(intent_features)  # [batch, intent_seq, hidden]
        context_h = self.context_proj(context_features)  # [batch, context_seq, hidden]
        
        # === Forward Attention: Intent attends to Context ===
        
        # Queries từ intent, Keys và Values từ context
        Q = self.q_proj(intent_h).view(batch_size, intent_seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(context_h).view(batch_size, context_seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(context_h).view(batch_size, context_seq_len, self.num_heads, self.head_dim)
        
        # Transpose cho multi-head attention: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, num_heads, intent_seq, context_seq]
        
        # Apply context mask nếu có
        if context_mask is not None:
            # Expand mask cho multi-head attention
            expanded_mask = context_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, context_seq]
            attention_scores = attention_scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Softmax over context dimension
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        attended_context = torch.matmul(attention_probs, V)  # [batch, num_heads, intent_seq, head_dim]
        
        # Reshape và project
        attended_context = attended_context.transpose(1, 2).contiguous().view(
            batch_size, intent_seq_len, self.hidden_dim
        )
        
        enhanced_intent = self.intent_out_proj(attended_context)
        enhanced_intent = self.intent_norm(enhanced_intent + intent_h)  # Residual connection
        
        # === Reverse Attention: Context attends to Intent ===
        
        # Queries từ context, Keys và Values từ intent
        Q_rev = self.q_proj_rev(context_h).view(batch_size, context_seq_len, self.num_heads, self.head_dim)
        K_rev = self.k_proj_rev(intent_h).view(batch_size, intent_seq_len, self.num_heads, self.head_dim)
        V_rev = self.v_proj_rev(intent_h).view(batch_size, intent_seq_len, self.num_heads, self.head_dim)
        
        Q_rev = Q_rev.transpose(1, 2)
        K_rev = K_rev.transpose(1, 2)
        V_rev = V_rev.transpose(1, 2)
        
        # Compute attention scores
        attention_scores_rev = torch.matmul(Q_rev, K_rev.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, num_heads, context_seq, intent_seq]
        
        # Softmax over intent dimension
        attention_probs_rev = F.softmax(attention_scores_rev, dim=-1)
        attention_probs_rev = self.dropout(attention_probs_rev)
        
        # Apply attention to values
        attended_intent = torch.matmul(attention_probs_rev, V_rev)  # [batch, num_heads, context_seq, head_dim]
        
        # Reshape và project
        attended_intent = attended_intent.transpose(1, 2).contiguous().view(
            batch_size, context_seq_len, self.hidden_dim
        )
        
        enhanced_context = self.context_out_proj(attended_intent)
        enhanced_context = self.context_norm(enhanced_context + context_h)  # Residual connection
        
        return enhanced_intent, enhanced_context


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion mechanism với learnable gating
    
    Dynamically determines optimal fusion weights based on input characteristics
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Gating network để determine fusion weights
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 2 weights: intent và context
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation layers
        self.intent_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.context_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self,
                intent_features: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion forward pass
        
        Args:
            intent_features: Intent features [batch_size, seq_len, feature_dim]  
            context_features: Context features [batch_size, seq_len, feature_dim]
            
        Returns:
            Fused features [batch_size, seq_len, feature_dim]
        """
        
        # Pool features để get global representations
        intent_pooled = torch.mean(intent_features, dim=1)  # [batch, feature_dim]
        context_pooled = torch.mean(context_features, dim=1)  # [batch, feature_dim]
        
        # Compute gating weights
        combined_pooled = torch.cat([intent_pooled, context_pooled], dim=-1)  # [batch, feature_dim * 2]
        gate_weights = self.gate_network(combined_pooled)  # [batch, 2]
        
        # Transform features
        transformed_intent = self.intent_transform(intent_features)
        transformed_context = self.context_transform(context_features)
        
        # Apply adaptive weights
        intent_weight = gate_weights[:, 0:1].unsqueeze(1)  # [batch, 1, 1]
        context_weight = gate_weights[:, 1:2].unsqueeze(1)  # [batch, 1, 1]
        
        weighted_intent = transformed_intent * intent_weight
        weighted_context = transformed_context * context_weight
        
        # Final fusion
        concatenated = torch.cat([weighted_intent, weighted_context], dim=-1)
        fused = self.fusion_layer(concatenated)
        
        return self.layer_norm(fused)


class HistoryAwareFusion(nn.Module):
    """
    Complete fusion mechanism combining all components
    
    Pipeline:
    1. Cross-modal attention between intent và context
    2. Adaptive fusion với gating mechanism
    3. Output projections cho downstream tasks
    """
    
    def __init__(self,
                 intent_dim: int = 768,
                 context_dim: int = 768,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 fusion_hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.intent_dim = intent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            intent_dim=intent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Adaptive fusion
        self.adaptive_fusion = AdaptiveFusion(
            feature_dim=hidden_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout
        )
        
        # Final output projections
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                intent_features: torch.Tensor,
                context_features: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        Complete fusion forward pass
        
        Args:
            intent_features: Intent features từ BERT [batch_size, seq_len, intent_dim]
            context_features: Context features từ GNN [batch_size, num_nodes, context_dim]
            context_mask: Mask cho context nodes [batch_size, num_nodes]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - 'fused_features': Final fused representations
            - 'enhanced_intent': Intent features after cross-attention
            - 'enhanced_context': Context features after cross-attention  
            - 'attention_weights': (optional) Cross-attention weights
        """
        
        # Cross-modal attention
        enhanced_intent, enhanced_context = self.cross_attention(
            intent_features, context_features, context_mask
        )
        
        # Adaptive fusion
        # Note: Need to handle sequence length differences
        batch_size = enhanced_intent.size(0)
        intent_seq_len = enhanced_intent.size(1)
        context_seq_len = enhanced_context.size(1)
        
        if intent_seq_len != context_seq_len:
            # Pool context features để match intent sequence length
            if context_seq_len > intent_seq_len:
                # Average pooling
                pooled_context = F.adaptive_avg_pool1d(
                    enhanced_context.transpose(1, 2), intent_seq_len
                ).transpose(1, 2)
            else:
                # Interpolation
                pooled_context = F.interpolate(
                    enhanced_context.transpose(1, 2), size=intent_seq_len, mode='linear', align_corners=False
                ).transpose(1, 2)
        else:
            pooled_context = enhanced_context
        
        # Adaptive fusion
        fused_features = self.adaptive_fusion(enhanced_intent, pooled_context)
        
        # Final projection
        final_features = self.final_projection(fused_features)
        final_features = self.layer_norm(final_features)
        
        # Prepare output
        output = {
            'fused_features': final_features,
            'enhanced_intent': enhanced_intent,
            'enhanced_context': enhanced_context
        }
        
        if return_attention_weights:
            # TODO: Extract attention weights từ cross_attention if needed
            output['attention_weights'] = None
        
        return output


# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing Fusion Mechanism...")
    
    # Test parameters
    batch_size = 2
    intent_seq_len = 20
    context_seq_len = 50
    intent_dim = 768
    context_dim = 768
    hidden_dim = 768
    
    # Create test data
    intent_features = torch.randn(batch_size, intent_seq_len, intent_dim)
    context_features = torch.randn(batch_size, context_seq_len, context_dim)
    context_mask = torch.ones(batch_size, context_seq_len, dtype=torch.bool)
    # Mask out some context nodes
    context_mask[:, -10:] = False
    
    # Test CrossModalAttention
    print("Testing CrossModalAttention...")
    cross_attn = CrossModalAttention(
        intent_dim=intent_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_heads=12
    )
    
    enhanced_intent, enhanced_context = cross_attn(intent_features, context_features, context_mask)
    print(f"  Intent features: {intent_features.shape} -> {enhanced_intent.shape}")
    print(f"  Context features: {context_features.shape} -> {enhanced_context.shape}")
    
    # Test AdaptiveFusion
    print("\nTesting AdaptiveFusion...")
    adaptive_fusion = AdaptiveFusion(feature_dim=hidden_dim)
    
    # Use same sequence length cho testing
    test_intent = enhanced_intent[:, :intent_seq_len]
    test_context = enhanced_context[:, :intent_seq_len]
    
    fused = adaptive_fusion(test_intent, test_context)
    print(f"  Input shapes: {test_intent.shape}, {test_context.shape}")
    print(f"  Fused output: {fused.shape}")
    
    # Test complete HistoryAwareFusion
    print("\nTesting HistoryAwareFusion...")
    fusion_model = HistoryAwareFusion(
        intent_dim=intent_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_heads=12
    )
    
    output = fusion_model(intent_features, context_features, context_mask)
    
    print(f"  Input intent: {intent_features.shape}")
    print(f"  Input context: {context_features.shape}")
    print(f"  Fused features: {output['fused_features'].shape}")
    print(f"  Enhanced intent: {output['enhanced_intent'].shape}")
    print(f"  Enhanced context: {output['enhanced_context'].shape}")
    
    # Test parameter count
    total_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test gradient flow
    loss = output['fused_features'].sum()
    loss.backward()
    print(f"  Gradient check: OK")
    
    print("✅ Fusion Mechanism testing completed!")