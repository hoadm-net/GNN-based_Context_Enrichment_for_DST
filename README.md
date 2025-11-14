git clone <repository_url>
# GNN-based Context Enrichment for Dialogue State Tracking

MultiWOZ dialogues contain long context windows and mixed-domain slot dependencies. This project explores a history-aware DST pipeline that combines transformer intent modelling with graph-based context reasoning so the belief state can be updated turn by turn without replaying the entire dialogue.

## Concept Overview

- **Goal**: track joint belief states across five MultiWOZ domains (hotel, restaurant, attraction, train, taxi) using only the structured data already supplied in `data/processed_graph/`.
- **Approach**: pair a BERT intent encoder with a multi-level graph over dialogue history (domain, schema, value, turn nodes). Features from both branches are fused and converted into delta predictions (keep/add/update/remove) so the tracker only reasons about changed slots each turn.
- **Result**: the current pipeline reaches test JGA ≈ 0.68 using the dataset version shipped in this repository.

## Model Architecture

The tracker follows a staged pipeline that incrementally enriches dialogue context before producing per-slot belief updates. Each block contributes a distinct type of signal:

### 1. Utterance Intent Encoder
- Backbone: `bert-base-uncased` (frozen lower layers, trainable projection).
- Input: the current user utterance plus the preceding system turn (concatenated with a separator token).
- Output: token embeddings and two pooled vectors (CLS token and learned attention pooling) representing utterance intent.
- Additional heads: a lightweight convolutional contrastive head encourages separation between intents that trigger different slot updates.

### 2. Multi-level Dialogue Graph Builder
- Construction handled by `src/data/graph_builders/` using the processed dataset (up to 20 past turns).
- Node sets:
  - **Domain Nodes** – one per active domain (hotel, restaurant, attraction, train, taxi).
  - **Schema Nodes** – all slots defined in `slot_meta.json`.
  - **Value Nodes** – high-support categorical values (e.g., price ranges, parking options).
  - **Turn Nodes** – each historical turn with utterance metadata and previous belief deltas.
- Edge families:
  - Domain↔Schema (slot-domain membership)
  - Schema↔Value (value admissibility)
  - Turn↔Schema (slot mentioned/updated in a turn)
  - Turn↔Turn (temporal order with exponential decay weights)
  - Schema↔Schema (manually curated correlations, e.g., `hotel-area`↔`restaurant-area`).
- Initialization:
  - Domain/value embeddings learned from scratch.
  - Schema embeddings seeded from averaged utterance token embeddings that mention the slot during preprocessing.
  - Turn embeddings aggregate TF-IDF bag-of-words and previous delta signatures.

### 3. Heterogeneous GNN Encoder
- Two stacked relational attention layers (`heterogeneous_gnn.py`).
- Message passing is type-specific: each edge family has its own projection matrix and attention parameters.
- Temporal reasoning is injected by gating messages along Turn↔Turn edges with a learned decay derived from time distance.
- Output: slot-centric context embeddings (each slot receives messages from all connected domains, values, and relevant turns) and pooled graph summaries.

### 4. Cross-modal Fusion
- Component: `advanced_fusion.py`.
- Steps:
  1. **Intent→Graph Attention** – intent tokens attend over slot embeddings to highlight slots implied by the utterance.
  2. **Graph→Intent Attention** – slot embeddings attend back over intent tokens to capture lexical grounding.
  3. **Gated Fusion** – the two attended vectors plus the original slot embedding are concatenated and passed through a gating MLP producing a fused slot representation.
- Regularisation: dropout + residual connections ensure stability when either branch carries weak signal.

### 5. Delta Prediction Heads
- Implemented in `classification_delta_heads.py` and wrapped inside the model.
- For every slot we predict:
  - **Operation logits** over {KEEP, ADD, UPDATE, REMOVE}.
  - **Value existence logit** (whether a value should be asserted at all).
  - **Special values** logits for `none` and `dontcare`.
  - **Categorical value classifier** when the slot’s vocabulary is provided in `slot_meta.json`.
- Slots without closed vocabularies bypass the categorical classifier; their updated value is expected to be copied externally (future work).
- A `DeltaTargetComputer` converts previous/current belief states into supervision signals so that training focuses on changed slots only.

### 6. Belief State Reconstruction
- During inference, predictions are converted into deltas and applied to the previous belief state maintained by the dataloader.
- Operation decoding is deterministic: KEEP leaves the slot untouched, ADD/UPDATE write the predicted value (or `dontcare`), REMOVE deletes the slot.
- The reconstructed state is returned with auxiliary attention maps for inspection.

## Repository Layout (Active Files)

```
GNN-based_Context_Enrichment_for_DST/
├── data/
│   ├── raw/                    # Original MultiWOZ resources (unchanged)
│   └── processed_graph/        # Preprocessed dataset used in this pipeline
├── src/
│   ├── data/                   # Graph builders and preprocessing helpers
│   └── models/                 # BERT encoder, GNN layers, fusion, delta heads
├── checkpoints/graph_dst/      # Saved weights; best model at epoch 8
├── results/graph_dst/          # Metrics and prediction dumps for dev/test
├── train_graph_dst.py          # Training entry point for the current pipeline
├── validate_graph_dst.py       # Evaluation entry point for dev/test splits
├── analysis_summary.json       # Slot-level error analysis for latest test run
├── requirements.txt
└── README.md
```

## Using the Pipeline

### Training

The script expects the processed graph dataset and ontology already present under `data/`. It can start from scratch or resume a checkpoint.

```bash
python train_graph_dst.py \
  --epochs 10                 # fresh run length
# Optional resume run:
python train_graph_dst.py \
  --resume checkpoints/graph_dst/best_model.pt \
  --additional-epochs 3
```

Training logs show turn-level loss and dev JGA after each epoch. The best-performing weights are always written to `checkpoints/graph_dst/best_model.pt`.

### Validation / Test

`validate_graph_dst.py` reloads a checkpoint, runs the requested split, and writes fresh metrics plus JSONL predictions under `results/graph_dst/` (or a custom directory via `--results_dir`).

```bash
python validate_graph_dst.py --split dev  --checkpoint checkpoints/graph_dst/best_model.pt
python validate_graph_dst.py --split test --checkpoint checkpoints/graph_dst/best_model.pt
```

Each evaluation reports joint goal accuracy and average loss so results stay comparable across runs.

## Next Directions

- **Value Normalisation**: smooth spelling variants (guest house vs guesthouse, night club vs nightclub) before scoring to recover JGA lost to formatting differences.
- **Entity Recall**: adjust history fusion or add post-processing to reduce missed carry-over for restaurant/hotel names.
- **Resume Training**: once value vocab issues are resolved, resume the best checkpoint with a lower learning rate to test whether more epochs help.

This README reflects the active codepath only; legacy experiments and unused utilities were removed to keep the project focused on the pipeline above.