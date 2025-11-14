"""
Configuration Management for History-Aware GraphDST

Chứa tất cả hyperparameters, model settings, và training configurations
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Intent Encoder
    bert_model_name: str = "bert-base-uncased"
    intent_hidden_dim: int = 768
    freeze_bert: bool = False
    
    # GNN Configuration  
    gnn_hidden_dim: int = 768
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1
    num_node_types: int = 6  # Turn, BeliefState, SlotValue, Domain, Slot, Value
    
    # Fusion Layer
    fusion_hidden_dim: int = 768
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    
    # Prediction Heads
    num_domains: int = 5  # Hotel, Restaurant, Attraction, Train, Taxi
    num_slots: int = 30   # Total slots across all domains
    max_value_length: int = 50


@dataclass
class TrainingConfig:
    """Training process configuration"""
    # Basic training settings
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Gradient settings
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Loss weights (legacy)
    domain_loss_weight: float = 0.1  # Reduced since it's auxiliary
    slot_loss_weight: float = 1.0     # Legacy compatibility
    value_loss_weight: float = 1.0    # Legacy compatibility
    
    # Delta DST loss weights
    operation_loss_weight: float = 2.0      # Primary loss - operation prediction
    value_existence_loss_weight: float = 1.0  # Whether slot has value
    none_loss_weight: float = 0.5           # Special "none" value
    dontcare_loss_weight: float = 0.5       # Special "dontcare" value
    span_loss_weight: float = 0.3           # Span extraction (future)
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Paths
    data_dir: str = "data/processed"
    raw_data_dir: str = "data/raw"
    slot_meta_path: str = "data/processed/slot_meta.json"
    
    # Text processing
    max_sequence_length: int = 512
    max_history_length: int = 10  # Number of previous turns to include
    max_utterance_length: int = 50
    
    # Graph construction
    max_graph_nodes: int = 200
    include_schema_graph: bool = True
    schema_connection_threshold: float = 0.5


@dataclass
class ExperimentConfig:
    """Experiment and logging configuration"""
    # Experiment tracking
    experiment_name: str = "history_aware_graphdst"
    run_name: Optional[str] = None
    use_wandb: bool = True
    wandb_project: str = "dst-experiments"
    
    # Output directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    results_dir: str = "outputs/results"
    
    # Evaluation
    eval_every_n_epochs: int = 1
    save_predictions: bool = True
    compute_per_domain_metrics: bool = True


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Device settings
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__,
            'device': self.device,
            'num_workers': self.num_workers,
            'seed': self.seed
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Update experiment config
        if 'experiment' in config_dict:
            for key, value in config_dict['experiment'].items():
                if hasattr(config.experiment, key):
                    setattr(config.experiment, key, value)
        
        # Update other fields
        for key in ['device', 'num_workers', 'seed']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create output directories
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(self.experiment.checkpoint_dir, exist_ok=True)
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.results_dir, exist_ok=True)


# Default configurations for different scenarios
def get_debug_config() -> Config:
    """Configuration for debugging with small data"""
    config = Config()
    config.training.batch_size = 4
    config.training.num_epochs = 2
    config.data.max_history_length = 3
    config.data.max_graph_nodes = 50
    config.experiment.use_wandb = False
    return config


def get_fast_config() -> Config:
    """Configuration for fast training"""
    config = Config()
    config.training.batch_size = 32
    config.training.num_epochs = 20
    config.training.learning_rate = 5e-5
    config.model.gnn_num_layers = 2
    return config


def get_full_config() -> Config:
    """Configuration for full training"""
    config = Config()
    config.training.batch_size = 16
    config.training.num_epochs = 50
    config.training.learning_rate = 2e-5
    config.model.gnn_num_layers = 3
    return config


if __name__ == "__main__":
    # Example usage
    config = get_full_config()
    print("Full configuration:")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"GNN layers: {config.model.gnn_num_layers}")
    
    # Save example config
    config.save_to_file("outputs/example_config.json")
    print("Configuration saved to outputs/example_config.json")