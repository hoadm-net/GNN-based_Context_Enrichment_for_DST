"""
Validation script for Multi-Level GNN-based DST model.

Loads a trained checkpoint, runs evaluation on the dev or test split,
computes key metrics, and writes prediction logs for inspection.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.updated_history_aware_graphdst import UpdatedHistoryAwareGraphDST
from src.data.graph_dataloader import GraphDSTDataset, collate_fn_graph
from train_graph_dst import compute_delta_loss


def load_datasets(data_dir: str, split: str, batch_size: int) -> Dict[str, Any]:
    """Prepare train (for vocab sharing) and evaluation dataloaders."""
    slot_meta_path = os.path.join(data_dir, "slot_meta.json")
    train_dataset = GraphDSTDataset(
        data_path=os.path.join(data_dir, "train_instances.json"),
        slot_meta_path=slot_meta_path
    )

    split_file = os.path.join(data_dir, f"{split}_instances.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    eval_dataset = GraphDSTDataset(
        data_path=split_file,
        slot_meta_path=slot_meta_path
    )

    # Share vocabularies so the evaluation dataset matches training mappings
    eval_dataset.slot2idx = train_dataset.slot2idx
    eval_dataset.idx2slot = train_dataset.idx2slot
    eval_dataset.value2idx = train_dataset.value2idx
    eval_dataset.idx2value = train_dataset.idx2value
    eval_dataset.slot_value_vocab = train_dataset.slot_value_vocab

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_graph
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "eval_loader": eval_loader
    }


def build_model(train_dataset: GraphDSTDataset,
                slot_meta_path: str,
                ontology_path: str,
                device: torch.device,
                fusion_type: str = "multimodal") -> UpdatedHistoryAwareGraphDST:
    """Instantiate model and load ontology using training vocabularies."""
    model = UpdatedHistoryAwareGraphDST(
        hidden_dim=768,
        fusion_dim=768,
        num_domains=5,
        max_history_turns=20,
        num_gnn_layers=2,
        num_heads=8,
        dropout=0.1,
        fusion_type=fusion_type
    )

    model.setup_ontology(
        slot_meta_path=slot_meta_path,
        ontology_path=ontology_path,
        slot_value_vocab=train_dataset.slot_value_vocab
    )

    model.to(device)
    return model


def evaluate_split(model: UpdatedHistoryAwareGraphDST,
                   eval_loader: DataLoader,
                   device: torch.device,
                   results_dir: Path,
                   split: str) -> Dict[str, Any]:
    """Run evaluation and persist metrics + prediction logs."""
    model.eval()
    metrics = {
        "total_instances": 0,
        "total_correct": 0,
        "total_loss": 0.0
    }

    logs = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating ({split})"):
            batch_size = batch["batch_size"]
            for i in range(batch_size):
                utterance = batch["utterances"][i]
                dialogue_history = batch["dialogue_histories"][i]
                previous_belief = batch["previous_belief_states"][i]
                current_target = batch["current_belief_states"][i]

                output = model(
                    utterance=utterance,
                    dialogue_history=dialogue_history,
                    previous_belief_state=previous_belief,
                    return_attention=False
                )

                loss = compute_delta_loss(
                    model_output=output,
                    model=model,
                    previous_belief_state=previous_belief,
                    current_belief_state=current_target
                ).item()

                predicted_belief = output.belief_state

                metrics["total_instances"] += 1
                metrics["total_loss"] += loss
                if predicted_belief == current_target:
                    metrics["total_correct"] += 1

                logs.append({
                    "dialogue_id": batch["dialogue_ids"][i],
                    "turn_id": batch["turn_ids"][i],
                    "user_utterance": utterance,
                    "system_response": batch["system_responses"][i],
                    "previous_belief_state": previous_belief,
                    "true_belief_state": current_target,
                    "predicted_belief_state": predicted_belief
                })

    total_instances = max(metrics["total_instances"], 1)
    avg_loss = metrics["total_loss"] / total_instances
    jga = metrics["total_correct"] / total_instances

    metrics_payload = {
        "split": split,
        "instances": total_instances,
        "avg_loss": avg_loss,
        "joint_goal_accuracy": jga
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics
    metrics_path = results_dir / f"metrics_{split}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Write prediction logs (JSON Lines)
    log_path = results_dir / f"predictions_{split}.jsonl"
    with log_path.open("w") as f:
        for record in logs:
            f.write(json.dumps(record) + "\n")

    return {
        "metrics": metrics_payload,
        "metrics_path": str(metrics_path),
        "log_path": str(log_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Validate GraphDST model on dev/test splits")
    parser.add_argument("--split", default="dev", choices=["dev", "val", "test"],
                        help="Dataset split to evaluate (aliases: dev==val)")
    parser.add_argument("--checkpoint", default="checkpoints/graph_dst/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="data/processed_graph",
                        help="Directory containing processed graph data")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--results_dir", default="results/graph_dst",
                        help="Directory to store evaluation artifacts")
    args = parser.parse_args()

    split_map = {"val": "dev", "dev": "dev", "test": "test"}
    split_key = split_map[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    data_bundle = load_datasets(args.data_dir, split_key, args.batch_size)
    train_dataset = data_bundle["train_dataset"]
    eval_loader = data_bundle["eval_loader"]

    # Build model and load checkpoint
    slot_meta_path = os.path.join(args.data_dir, "slot_meta.json")
    ontology_path = os.path.join(os.path.abspath(os.path.join(args.data_dir, os.pardir)), "raw/ontology.json")

    model = build_model(
        train_dataset=train_dataset,
        slot_meta_path=slot_meta_path,
        ontology_path=ontology_path,
        device=device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_split(
        model=model,
        eval_loader=eval_loader,
        device=device,
        results_dir=Path(args.results_dir),
        split=split_key
    )

    print("\nEvaluation complete:")
    print(json.dumps(results["metrics"], indent=2))
    print(f"Metrics saved to: {results['metrics_path']}")
    print(f"Prediction log saved to: {results['log_path']}")


if __name__ == "__main__":
    main()
