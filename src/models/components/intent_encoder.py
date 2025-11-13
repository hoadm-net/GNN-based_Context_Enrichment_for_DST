"""
Intent Encoder Module - BERT cho Current Utterance Only

Xử lý current user utterance để extract intent features,
tách biệt hoàn toàn với historical context processing.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Optional, Tuple


class IntentEncoder(nn.Module):
    """
    BERT-based intent encoder cho current utterance only
    
    Key advantages:
    - Focused processing: chỉ current utterance
    - Reduced complexity: shorter input sequences  
    - Cleaner intent signal: không bị nhiễu từ history
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(model_name)
        
        # Feature projection (if needed)
        bert_hidden_size = self.bert.config.hidden_size
        if bert_hidden_size != hidden_dim:
            self.feature_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.feature_projection = nn.Identity()
            
        # Layer norm và dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass cho intent encoding
        
        Args:
            input_ids: [batch, seq_len] - tokenized current utterance
            attention_mask: [batch, seq_len] - attention mask
            
        Returns:
            Dict containing:
            - intent_features: [batch, hidden_dim] - CLS representation
            - token_features: [batch, seq_len, hidden_dim] - token representations
        """
        
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract features
        last_hidden_state = bert_outputs.last_hidden_state  # [batch, seq_len, bert_hidden]
        cls_token = last_hidden_state[:, 0]  # [batch, bert_hidden]
        
        # Project to target dimension
        intent_features = self.feature_projection(cls_token)  # [batch, hidden_dim]
        token_features = self.feature_projection(last_hidden_state)  # [batch, seq_len, hidden_dim]
        
        # Apply layer norm và dropout
        intent_features = self.dropout(self.layer_norm(intent_features))
        token_features = self.dropout(self.layer_norm(token_features))
        
        return {
            'intent_features': intent_features,
            'token_features': token_features,
            'attention_mask': attention_mask
        }
    
    def encode_utterances(self, 
                         utterances: List[str], 
                         tokenizer: BertTokenizer,
                         max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of utterances
        
        Args:
            utterances: List of current utterances
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Encoded features dictionary
        """
        
        # Tokenize utterances
        encoded = tokenizer(
            utterances,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass
        return self.forward(input_ids, attention_mask)


class IntentProcessor:
    """
    Utility class để process current utterances
    """
    
    def __init__(self, 
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def preprocess_utterance(self, utterance: str) -> str:
        """
        Clean và preprocess current utterance
        
        Args:
            utterance: Raw user utterance
            
        Returns:
            Cleaned utterance
        """
        
        # Basic cleaning
        utterance = utterance.strip().lower()
        
        # Remove extra whitespace
        utterance = ' '.join(utterance.split())
        
        return utterance
    
    def extract_current_utterance(self, dialog_turn: Dict) -> str:
        """
        Extract current user utterance từ dialog turn
        
        Args:
            dialog_turn: Dialog turn data
            
        Returns:
            Current user utterance
        """
        
        # Extract current utterance only (no history)
        current_utterance = dialog_turn.get('current_utterance', '')
        
        if not current_utterance:
            # Fallback to user utterance field
            current_utterance = dialog_turn.get('user_utterance', '')
            
        return self.preprocess_utterance(current_utterance)
    
    def tokenize_batch(self, utterances: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of current utterances
        
        Args:
            utterances: List of current utterances
            
        Returns:
            Tokenized batch
        """
        
        return self.tokenizer(
            utterances,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )


# Example usage and testing
if __name__ == "__main__":
    # Test intent encoder
    print("🧪 Testing Intent Encoder...")
    
    # Create encoder
    intent_encoder = IntentEncoder()
    intent_processor = IntentProcessor()
    
    # Test utterances
    test_utterances = [
        "I need parking",
        "Find me a cheap hotel",
        "I want to book a table for 2",
        "What time does the train leave?"
    ]
    
    print(f"Test utterances: {test_utterances}")
    
    # Process utterances
    processed_utterances = [
        intent_processor.preprocess_utterance(utt) 
        for utt in test_utterances
    ]
    
    print(f"Processed utterances: {processed_utterances}")
    
    # Encode utterances
    with torch.no_grad():
        results = intent_encoder.encode_utterances(
            processed_utterances, 
            intent_processor.tokenizer
        )
    
    print(f"Intent features shape: {results['intent_features'].shape}")
    print(f"Token features shape: {results['token_features'].shape}")
    print(f"Attention mask shape: {results['attention_mask'].shape}")
    
    # Test với dialog turn format
    sample_dialog_turn = {
        'current_utterance': "I need a hotel with free wifi",
        'turn_id': 3,
        'previous_belief_state': {'hotel-internet': 'yes'}
    }
    
    current_utt = intent_processor.extract_current_utterance(sample_dialog_turn)
    print(f"Extracted current utterance: '{current_utt}'")
    
    print("✅ Intent Encoder testing completed!")