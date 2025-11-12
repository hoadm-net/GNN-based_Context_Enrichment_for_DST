# GNN-based Context Enrichment for Dialogue State Tracking

A comprehensive research project implementing Graph Neural Network-based context enrichment for Dialogue State Tracking (DST) on the MultiWOZ 2.4 dataset.

## 🎯 Project Overview

This project aims to enhance Dialogue State Tracking performance by leveraging Graph Neural Networks to model complex relationships between dialogue context, domain knowledge, and state transitions. Built upon state-of-the-art DST approaches including STAR, ASSIST, and MetaASSIST.

## 📊 Dataset

- **MultiWOZ 2.4**: Multi-domain task-oriented dialogue dataset
- **Domains**: Hotel, Restaurant, Attraction, Train, Taxi
- **Statistics**: 
  - Training: 54,984 instances (78.9%)
  - Development: 7,365 instances (10.6%) 
  - Test: 7,368 instances (10.6%)
  - Total Slots: 30 across 5 domains

## 🏗️ Project Structure

```
GNN-based_Context_Enrichment_for_DST/
├── data/
│   ├── raw/                    # Raw MultiWOZ 2.4 files
│   └── processed/              # Preprocessed training instances
├── src/
│   ├── data/
│   │   ├── download_data.py    # Dataset downloader
│   │   └── preprocess.py       # Data preprocessing pipeline
│   ├── models/                 # Model implementations
│   ├── evaluation/
│   │   └── evaluator.py        # Evaluation framework
│   └── utils/
│       └── data_loader.py      # Data loading utilities
├── notebooks/                  # Jupyter notebooks for analysis
├── results/                    # Experimental results
├── test_pipeline.py           # Pipeline testing script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd GNN-based_Context_Enrichment_for_DST

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download MultiWOZ 2.4 dataset
python src/data/download_data.py

# Preprocess the data
python src/data/preprocess.py

# Test the pipeline
python test_pipeline.py
```

### 3. Data Loading Example

```python
from src.utils.data_loader import DSTDataLoader

# Initialize data loader
data_loader = DSTDataLoader(
    data_dir="data/processed",
    slot_meta_path="data/processed/slot_meta.json",
    batch_size=16
)

# Get training data loader
train_loader = data_loader.get_dataloader('train', shuffle=True)

# Iterate through batches
for batch in train_loader:
    dialogue_ids = batch['dialogue_ids']
    input_texts = batch['input_texts']
    belief_states = batch['belief_states']
    # Your training code here...
```

### 4. Evaluation Example

```python
from src.evaluation.evaluator import DSTEvaluator

# Load slot metadata
with open("data/processed/slot_meta.json", 'r') as f:
    slot_meta = json.load(f)['slot_meta']

# Initialize evaluator
evaluator = DSTEvaluator(slot_meta)

# Evaluate predictions
metrics = evaluator.evaluate_predictions(predictions, ground_truth)

# Print results
evaluator.print_evaluation_results(metrics, model_name="Your Model")
```

## 📈 Evaluation Metrics

The evaluation framework implements standard DST metrics:

- **Joint Goal Accuracy (JGA)**: Percentage of turns where all slot values are predicted correctly
- **Slot Accuracy**: Average accuracy across all slots
- **Turn Accuracy**: Accuracy of predicting slot changes in each turn
- **Final Joint Accuracy**: JGA computed only on final turns of dialogues
- **Per-domain Accuracy**: JGA broken down by dialogue domains

## 🔧 Data Processing Pipeline

### Text Normalization
- Lowercasing and whitespace normalization
- Phone number and postcode standardization
- Time and price pattern replacement
- Contraction and punctuation handling

### Instance Creation
- Dialogue history tracking (last 10 turns)
- Belief state conversion to slot-value pairs
- Turn-level labeling for state changes
- Domain-specific processing

### Quality Assurance
- ASCII text validation
- Turn length filtering (max 50 words)
- Dialogue structure validation
- Missing data handling

## 📚 Key Components

### DSTDataset
PyTorch Dataset class for DST training with:
- Configurable dialogue history length
- Flexible input text formatting
- Belief state normalization
- Support for tokenized inputs

### DSTEvaluator
Comprehensive evaluation framework with:
- Multiple accuracy metrics
- Per-domain analysis
- Detailed error reporting
- Statistical significance testing

### MultiWOZPreprocessor
Robust preprocessing pipeline with:
- Text normalization following MultiWOZ standards
- Belief state extraction and formatting
- Data split management
- Quality filtering and validation

## 🎯 Research Directions

This framework supports research in:
- **Graph Neural Networks**: Modeling dialogue structure and domain relationships
- **Context Enrichment**: Leveraging dialogue history and domain knowledge
- **Multi-domain Transfer**: Cross-domain learning and adaptation
- **Noise-robust Learning**: Handling annotation errors and inconsistencies

## 🔬 Baseline Models

The framework is designed to support various DST architectures:
- **BERT-based models**: Fine-tuned language models
- **STAR**: Slot self-attention mechanisms
- **ASSIST**: Noise-robust training approaches  
- **MetaASSIST**: Meta-learning for robust DST
- **GNN-enhanced models**: Graph-based context modeling

## 📊 Benchmark Results

MultiWOZ 2.4 benchmark results for reference:

| Model | Joint Goal Accuracy |
|-------|-------------------|
| SUMBT | 61.86% |
| STAR | 73.62% |
| TripPy | 64.75% |
| ASSIST | 79.41% |
| MetaASSIST | 80.10% |

## 🛠️ Development

### Running Tests
```bash
# Test complete pipeline
python test_pipeline.py

# Test specific components
python src/utils/data_loader.py --data_dir data/processed
python src/evaluation/evaluator.py --help
```

### Adding New Models
1. Create model class in `src/models/`
2. Implement training script
3. Use `DSTDataLoader` for data loading
4. Use `DSTEvaluator` for evaluation
5. Save results in `results/` directory

### Data Format
Training instances are stored in JSON format:
```json
{
  "dialogue_id": "dialogue_name",
  "turn_id": 0,
  "user_utterance": "normalized user text",
  "system_response": "normalized system text", 
  "dialogue_history": "conversation context",
  "belief_state": [["slot-name", "value"], ...],
  "domains": ["hotel", "restaurant"],
  "is_last_turn": false
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MultiWOZ Team** for the comprehensive dialogue dataset
- **DST-STAR** authors for slot self-attention mechanisms
- **DST-ASSIST** authors for noise-robust training approaches
- **DST-MetaASSIST** authors for meta-learning frameworks

## 📧 Contact

For questions and collaborations, please reach out through:
- GitHub Issues for technical questions
- Project discussions for research collaborations

---

**Ready to enhance dialogue state tracking with graph neural networks!** 🚀