# History-Aware GraphDST - Kiến Trúc và Implementation

## 1. Mô Tả Kiến Trúc Tổng Quan

**History-Aware GraphDST** là một kiến trúc mới cho Dialog State Tracking kết hợp **BERT** cho intent understanding và **Graph Neural Networks** cho historical context modeling. Khác với các approach truyền thống, chúng tôi tách biệt việc xử lý current utterance và historical context để tận dụng tối đa strengths của từng component.

### 1.1 Đặc Điểm Chính

• **Intent-Focused Processing**: BERT chỉ xử lý current utterance để focused intent understanding
• **History-Centric Graph**: GNN model lịch sử dialog như dynamic graph structure  
• **Dynamic Graph Construction**: Graph nodes được tạo từ previous belief states
• **Cross-Modal Fusion**: Intent features và context features được kết hợp qua attention mechanism
• **Modular Architecture**: Các component độc lập, dễ debug và maintain

### 1.2 Thành Phần Chính

```
Input: Current Utterance + Dialog History + Schema
    ↓
┌─────────────────┐         ┌─────────────────────────────┐
│   BERT BRANCH   │         │        GNN BRANCH           │
│                 │         │                             │
│Current Utterance│         │ History + Schema            │
│        ↓        │         │           ↓                 │
│ Intent Features │         │ ┌─────────────────────────┐ │
│   [batch, 768]  │         │ │ History Graph Builder   │ │
└─────────────────┘         │ │ - Previous belief states│ │
                            │ │ - Turn-level nodes      │ │
                            │ │ - Temporal edges        │ │
                            │ └─────────────────────────┘ │
                            │           ↓                 │
                            │ ┌─────────────────────────┐ │
                            │ │ Schema Graph Builder    │ │
                            │ │ - Domain nodes          │ │
                            │ │ - Slot nodes            │ │
                            │ │ - Value nodes           │ │
                            │ └─────────────────────────┘ │
                            │           ↓                 │
                            │ ┌─────────────────────────┐ │
                            │ │ Graph Neural Network    │ │
                            │ │ - Schema-aware GCN      │ │
                            │ │ - Cross-domain GAT      │ │
                            │ │ - Temporal reasoning    │ │
                            │ └─────────────────────────┘ │
                            │           ↓                 │
                            │ Context Features            │
                            │ [batch, num_nodes, 768]     │
                            └─────────────────────────────┘
                                        ↓
            ┌───────────────────────────────────────────────────┐
            │              FUSION LAYER                         │
            │ Cross-Attention: Intent ↔ Context                │
            │ Output: Enhanced representation [batch, 768]     │
            └───────────────────────────────────────────────────┘
                                        ↓
            ┌───────────────────────────────────────────────────┐
            │            PREDICTION HEADS                       │
            │ - Domain classification (5 domains)              │
            │ - Slot activation (30 slots)                     │
            │ - Value prediction (categorical)                 │
            └───────────────────────────────────────────────────┘
```

### 1.3 Workflow Tổng Quan

1. **Intent Processing**: BERT encoder xử lý current utterance → intent features
2. **History Graph Construction**: Previous belief states → dynamic graph nodes
3. **Schema Graph Integration**: Static ontology → schema structure  
4. **Graph Neural Network**: Multi-layer GNN processing unified graph
5. **Cross-Modal Fusion**: Intent features ↔ Context features via attention
6. **Multi-task Prediction**: Domain/Slot/Value prediction heads
7. **Output**: Belief state delta (incremental changes only)

---

## 2. Kiến Trúc Mô Hình Chi Tiết

### 2.1 Intent Encoder Module

**BERT-based Intent Understanding**

```python
# Intent Encoder: BERT cho current utterance only
Input: current_utterance = "I need parking"
Model: bert-base-uncased
Output: intent_features = [batch, 768]  # CLS token representation

# Key advantages:
- Focused processing: chỉ current utterance
- Reduced complexity: shorter input sequences  
- Cleaner intent signal: không bị nhiễu từ history
```

**Features:**
• Model: `bert-base-uncased` (110M params)
• Input length: Typically 20-50 tokens (vs 512 in traditional approaches)
• Output: CLS token representation [batch, 768]
• Purpose: Pure intent understanding without historical noise

### 2.2 History Graph Builder Module

**Dynamic Graph Construction từ Dialog History**

```python
# History Graph Construction
class HistoryGraphBuilder:
    def build_graph(self, dialog_history):
        nodes = []
        edges = []
        
        # Create nodes for each turn
        for turn_id, turn_data in enumerate(dialog_history):
            # Turn node
            turn_node = TurnNode(turn_id, turn_data.utterance)
            nodes.append(turn_node)
            
            # Belief state node
            if turn_data.belief_state:
                bs_node = BeliefStateNode(turn_id, turn_data.belief_state)
                nodes.append(bs_node)
                
                # Edge: Turn → Belief State
                edges.append(Edge(turn_node.id, bs_node.id, "generates"))
                
                # Individual slot-value nodes
                for slot, value in turn_data.belief_state.items():
                    sv_node = SlotValueNode(slot, value, turn_id)
                    nodes.append(sv_node)
                    
                    # Edge: Belief State → Slot-Value
                    edges.append(Edge(bs_node.id, sv_node.id, "contains"))
        
        return Graph(nodes, edges)
```

**Node Types:**
• **Turn Nodes**: Represent individual dialog turns
• **Belief State Nodes**: Complete belief state snapshots
• **Slot-Value Nodes**: Individual slot assignments

**Edge Types:**
• **Temporal**: Connect consecutive turns (turn_t → turn_t+1)
• **State Evolution**: Belief state transitions (bs_t → bs_t+1)  
• **Contains**: Belief state contains slot-values
• **Dependencies**: Related slots (hotel-area ↔ hotel-pricerange)

### 2.3 Schema Graph Builder Module

**Static Ontology Structure**

```python
# Schema Graph từ MultiWOZ ontology
Schema Structure:
├── Domain Nodes (5): [hotel, restaurant, attraction, train, taxi]
├── Slot Nodes (30): [hotel-area, hotel-price, restaurant-food, ...]
└── Value Nodes (~1700): [cheap, expensive, center, north, ...]

Edge Types:
├── Domain-Slot: hotel → hotel-area
├── Slot-Value: hotel-area → center  
└── Slot-Slot: hotel-area ↔ hotel-type (co-occurrence)
```

**Features:**
• Pre-trained embeddings cho domain/slot/value nodes
• Static structure (không thay đổi theo dialog)
• Rich semantic relationships
• Co-occurrence statistics từ training data

### 2.4 Graph Neural Network Layers

**Multi-layer GNN Architecture**

#### 2.4.1 Schema-Aware GCN Layer

```python
class SchemaAwareGCN(nn.Module):
    def forward(self, unified_graph):
        # Process different node types separately
        for node_type in ['turn', 'belief_state', 'slot_value', 'domain', 'slot', 'value']:
            node_features = unified_graph.get_nodes(node_type)
            
            # Message passing within node type
            messages = self.aggregate_messages(node_features, unified_graph.edges)
            
            # Update node features
            updated_features = self.update_function(node_features, messages)
            unified_graph.update_nodes(node_type, updated_features)
        
        return unified_graph
```

#### 2.4.2 Cross-Domain GAT Layer

```python
class CrossDomainGAT(nn.Module):
    def forward(self, unified_graph):
        # Multi-head attention across different domains
        for head in range(self.num_heads):
            # Compute attention scores
            attention_scores = self.compute_attention(
                unified_graph.node_features,
                unified_graph.edges
            )
            
            # Weighted message aggregation
            messages = self.aggregate_with_attention(
                unified_graph.node_features,
                attention_scores
            )
            
            # Update features
            head_outputs.append(messages)
        
        # Concatenate multi-head outputs
        return self.output_projection(torch.cat(head_outputs, dim=-1))
```

#### 2.4.3 Temporal Reasoning Layer

```python
class TemporalGNN(nn.Module):
    def forward(self, unified_graph):
        # Extract temporal sequences
        temporal_sequences = self.extract_sequences(unified_graph)
        
        # GRU processing for temporal dependencies
        for seq in temporal_sequences:
            gru_output, _ = self.gru(seq)
            self.update_temporal_features(unified_graph, gru_output)
        
        return unified_graph
```

### 2.5 Cross-Modal Fusion Layer

**Intent-Context Attention Mechanism**

```python
class CrossAttentionFusion(nn.Module):
    def forward(self, intent_features, context_features):
        # intent_features: [batch, 768]
        # context_features: [batch, num_nodes, 768]
        
        batch_size = intent_features.size(0)
        
        # Expand intent features for attention
        intent_expanded = intent_features.unsqueeze(1)  # [batch, 1, 768]
        
        # Multi-head attention
        attention_output, attention_weights = self.multihead_attention(
            query=intent_expanded,      # What we want to know
            key=context_features,       # Where to look
            value=context_features      # What to extract
        )
        
        # Combine intent with attended context
        fused_features = self.fusion_layer(
            torch.cat([intent_features, attention_output.squeeze(1)], dim=-1)
        )
        
        return fused_features, attention_weights
```

**Key Features:**
• Query: Intent features (what user wants now)
• Key/Value: Context features (historical knowledge)
• Output: Enhanced representation for prediction
• Interpretability: Attention weights show which history is relevant

### 2.6 Multi-Task Prediction Heads

**Domain, Slot, Value Prediction**

```python
class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim, slot_info):
        # Domain classification head
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)  # 5 domains
        )
        
        # Slot activation heads  
        self.slot_heads = nn.ModuleDict({
            slot: nn.Linear(hidden_dim, 2)  # binary: active/inactive
            for slot in slot_info['slot_names']
        })
        
        # Value prediction heads
        self.value_heads = nn.ModuleDict({
            slot: nn.Linear(hidden_dim, vocab_size)
            for slot, vocab_size in slot_info['categorical_slots'].items()
        })
    
    def forward(self, fused_features):
        predictions = {}
        
        # Domain predictions
        predictions['domains'] = torch.sigmoid(self.domain_head(fused_features))
        
        # Slot predictions
        predictions['slots'] = {
            slot: head(fused_features)
            for slot, head in self.slot_heads.items()
        }
        
        # Value predictions (only for active slots)
        predictions['values'] = {
            slot: head(fused_features)
            for slot, head in self.value_heads.items()
        }
        
        return predictions
```

### 2.7 Loss Function

**Multi-task Weighted Loss**

```python
def compute_loss(predictions, labels, loss_weights=None):
    if loss_weights is None:
        loss_weights = {'domain': 1.0, 'slot': 1.0, 'value': 1.0}
    
    losses = {}
    
    # 1. Domain loss (multi-label BCE)
    domain_loss = F.binary_cross_entropy(
        predictions['domains'],
        labels['domain_labels'].float()
    )
    losses['domain'] = domain_loss * loss_weights['domain']
    
    # 2. Slot activation losses
    slot_losses = []
    for slot in predictions['slots']:
        if f'{slot}_active' in labels:
            slot_loss = F.cross_entropy(
                predictions['slots'][slot],
                labels[f'{slot}_active'].long()
            )
            slot_losses.append(slot_loss)
    
    if slot_losses:
        losses['slot'] = torch.stack(slot_losses).mean() * loss_weights['slot']
    
    # 3. Value prediction losses (only for active slots)
    value_losses = []
    for slot in predictions['values']:
        if f'{slot}_value' in labels and labels[f'{slot}_active'].any():
            # Only compute loss for active slots
            active_mask = labels[f'{slot}_active'].bool()
            if active_mask.any():
                value_loss = F.cross_entropy(
                    predictions['values'][slot][active_mask],
                    labels[f'{slot}_value'][active_mask].long()
                )
                value_losses.append(value_loss)
    
    if value_losses:
        losses['value'] = torch.stack(value_losses).mean() * loss_weights['value']
    
    # Total loss
    losses['total'] = sum(losses.values())
    
    return losses
```

---

## 3. Implementation Details

### 3.1 Model Statistics

```
Total Parameters: ~140M
├── Intent Encoder (BERT): ~110M
├── History Graph Builder: ~5M  
├── Schema Graph Builder: ~2M
├── GNN Layers (3x): ~15M
├── Fusion Layer: ~3M
└── Prediction Heads: ~5M

Memory Usage:
├── Training: ~8-12 GB (batch_size=16)
├── Inference: ~2-4 GB
└── Graph Storage: ~100-500 MB per dialog

Performance Target:
├── Delta Accuracy: >90%
├── Joint Goal Accuracy: >58%
├── Training Time: ~4-6 hours (10 epochs, V100)
└── Inference Speed: ~150 examples/second
```

### 3.2 Key Advantages

**Compared to Traditional Approaches:**

| Aspect | Traditional GraphDST | History-Aware GraphDST |
|--------|---------------------|------------------------|
| BERT Input | Full context (512 tokens) | Current only (~30 tokens) |
| History Processing | Text-based via BERT | Graph-based via GNN |
| Context Modeling | Implicit in text | Explicit in graph structure |
| Interpretability | Limited | Graph attention weights |
| Efficiency | High memory usage | Reduced BERT overhead |
| Flexibility | Fixed schema only | Dynamic history + schema |

**Unique Innovations:**

• **History-Centric Design**: First to model dialog history as dynamic graph nodes
• **Intent-Context Separation**: Clean separation of current intent vs historical context
• **Cross-Modal Fusion**: Novel attention mechanism between intent and context
• **Modular Architecture**: Independent components for easy debugging/improvement

---

## 4. Training và Evaluation

### 4.1 Training Strategy

```python
# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        predictions = model(
            current_utterance=batch['current_utterance'],
            dialog_history=batch['dialog_history'], 
            schema=batch['schema']
        )
        
        # Compute loss
        losses = compute_loss(predictions, batch['labels'])
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 4.2 Evaluation Metrics

• **Delta Accuracy**: Percentage of turns với correct belief state changes
• **Joint Goal Accuracy**: Percentage of dialogs với all deltas correct
• **Slot F1**: Per-slot precision/recall/F1 scores
• **Domain F1**: Per-domain activation accuracy
• **Attention Analysis**: Qualitative analysis of attention weights

---

## 5. Tổng Kết

**History-Aware GraphDST** represents a paradigm shift in dialog state tracking:

1. **Intent-Focused Processing**: BERT specializes in current utterance understanding
2. **History-Centric Modeling**: GNN explicitly captures dialog evolution
3. **Cross-Modal Intelligence**: Attention-based fusion of intent and context
4. **Modular Design**: Independent, debuggable components
5. **Superior Performance**: Expected improvements in accuracy and efficiency

**Innovation Impact:**
- First architecture to model dialog history as dynamic graph structure
- Novel separation of intent understanding and context reasoning
- Practical improvements in memory efficiency and interpretability
- Foundation for future multi-modal dialog systems

---

**Document Version:** 2.0  
**Last Updated:** November 13, 2025  
**Author:** History-Aware GraphDST Development Team