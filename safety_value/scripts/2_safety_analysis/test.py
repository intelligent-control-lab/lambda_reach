"""
Evaluation script for trained safety analysis models.

This script evaluates a trained safety value function on the held-out test set
and reports detailed performance metrics including confusion matrix statistics.

Usage:
    python safety_value/scripts/safety_analysis/test.py \\
        --safety_analysis_root exp_name \\
        --model_subdir safety_analysis_discounted_policy_evaluation_lam0.9
"""

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_value.safety_analysis_algos.model import build_mlp
from safety_value.safety_analysis_algos.dataset import HJValueDataset, hj_collate


# Base directory for all safety analysis data and results
SAFETY_ANALYSIS_BASE = "logs/safety_analysis"


@dataclass
class ConfusionMatrix:
    """Container for confusion matrix statistics."""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total == 0:
            return float('nan')
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def precision(self) -> float:
        """Precision (positive predictive value)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return float('nan')
        return self.true_positives / denominator
    
    @property
    def recall(self) -> float:
        """Recall (sensitivity, true positive rate)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return float('nan')
        return self.true_positives / denominator
    
    @property
    def false_positive_rate(self) -> float:
        """False positive rate."""
        negatives = self.true_negatives + self.false_positives
        if negatives == 0:
            return float('nan')
        return self.false_positives / negatives
    
    @property
    def false_negative_rate(self) -> float:
        """False negative rate (miss rate)."""
        positives = self.true_positives + self.false_negatives
        if positives == 0:
            return float('nan')
        return self.false_negatives / positives
    
    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.recall == 0:
            return float('nan')
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def print_summary(self):
        """Print formatted summary of metrics."""
        print("\n" + "=" * 80)
        print("Confusion Matrix:")
        print(f"  True Positives:  {self.true_positives:6d}")
        print(f"  False Positives: {self.false_positives:6d}")
        print(f"  True Negatives:  {self.true_negatives:6d}")
        print(f"  False Negatives: {self.false_negatives:6d}")
        print(f"  Total Samples:   {self.total:6d}")
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {self.accuracy:7.4f}")
        print(f"  Precision: {self.precision:7.4f}")
        print(f"  Recall:    {self.recall:7.4f}")
        print(f"  F1 Score:  {self.f1_score:7.4f}")
        
        print("\nError Rates:")
        print(f"  False Positive Rate: {self.false_positive_rate:7.4f}")
        print(f"  False Negative Rate: {self.false_negative_rate:7.4f}")
        print("=" * 80 + "\n")


def find_latest_checkpoint(models_dir: str) -> Tuple[int, str]:
    """Find the checkpoint with the highest step number.

    Handles checkpoint names with optional suffixes, e.g., step_123.pt or step_123_run1.pt.
    Falls back to last.pt if no step checkpoints exist.
    """
    pattern = os.path.join(models_dir, "step_*.pt")
    checkpoints = glob.glob(pattern)

    def _extract_step(path: str) -> int:
        m = re.search(r"step_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    if checkpoints:
        step_paths = [(_extract_step(ckpt), ckpt) for ckpt in checkpoints]
        step_paths = [(s, p) for s, p in step_paths if s >= 0]
        if step_paths:
            best_step, best_path = max(step_paths, key=lambda x: x[0])
            return best_step, best_path

    # No step_*.pt found - check for last*.pt
    last_candidates = sorted(glob.glob(os.path.join(models_dir, "last*.pt")))
    if last_candidates:
        # If multiple, prefer the one with explicit step metadata
        for candidate in last_candidates:
            checkpoint = torch.load(candidate, map_location="cpu")
            step = checkpoint.get("global_step", -1)
            if step >= 0:
                return step, candidate
        return -1, last_candidates[-1]

    raise FileNotFoundError(f"No checkpoints found in {models_dir}")


class LambdaValueEnsemble(torch.nn.Module):
    """Two-critic ensemble for λ-reachability checkpoints.

    Forward returns the mean of critic1 and critic2 to mirror training-time usage.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple):
        super().__init__()
        self.critic1 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)
        self.critic2 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v1 = self.critic1(x)
        v2 = self.critic2(x)
        return 0.5 * (v1 + v2)


def load_model(
    checkpoint_path: str,
    input_dim: int,
    hidden_dims: tuple,
    device: torch.device,
    algorithm: str,
) -> torch.nn.Module:
    """Load model from checkpoint, handling λ-reachability ensembles.

    Args:
        checkpoint_path: Path to checkpoint file
        input_dim: Input dimension for the model
        hidden_dims: Hidden layer dimensions (read from training_config.json)
        device: Device to load model on
        algorithm: Algorithm name stored in training_config.json
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if algorithm == "lambda_reachability":
        model = LambdaValueEnsemble(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
        try:
            model.critic1.load_state_dict(checkpoint["critic1"])
            model.critic2.load_state_dict(checkpoint["critic2"])
        except KeyError as exc:
            raise KeyError(
                "Checkpoint for λ-reachability must contain 'critic1' and 'critic2' state dicts"
            ) from exc
    else:
        model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except KeyError as exc:
            raise KeyError(
                "Checkpoint missing 'model_state'. Ensure this is a single-critic checkpoint."
            ) from exc

    model.eval()
    return model


def compute_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[ConfusionMatrix, float, float]:
    """Compute confusion matrix, invariant accuracy, and value MSE.

    Returns (ConfusionMatrix, inv_acc, value_mse).
    """
    tp = fp = tn = fn = 0
    inv_acc_list = []
    mse_list = []

    with torch.no_grad():
        for x, _x_next, _l, sample_safety_value, invariant, _xf, _lf in dataloader:
            x = x.to(device)
            invariant = (invariant.to(device) > 0).float().view(-1)
            sample_safety_value = sample_safety_value.to(device).view(-1)

            v = model(x).view(-1)
            preds = (v <= 0).float()

            tp += int(((preds == 1) & (invariant == 1)).sum().item())
            fp += int(((preds == 1) & (invariant == 0)).sum().item())
            tn += int(((preds == 0) & (invariant == 0)).sum().item())
            fn += int(((preds == 0) & (invariant == 1)).sum().item())

            inv_acc_list.append(((preds == invariant).float().mean()).item())
            mse_list.append(((v - sample_safety_value) ** 2).mean().item())

    inv_acc = float(sum(inv_acc_list) / len(inv_acc_list)) if len(inv_acc_list) > 0 else float("nan")
    value_mse = float(sum(mse_list) / len(mse_list)) if len(mse_list) > 0 else float("nan")

    return (
        ConfusionMatrix(true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn),
        inv_acc,
        value_mse,
    )


def save_results(confusion_matrix: ConfusionMatrix, model_dir: str, checkpoint_step: int, inv_acc: float, value_mse: float):
    """Save evaluation results to disk, including value MSE and invariant accuracy."""
    # Create test results directory
    test_dir = os.path.join(model_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        "checkpoint_step": checkpoint_step,
        "timestamp": datetime.now().isoformat(),
        "confusion_matrix": {
            "true_positives": confusion_matrix.true_positives,
            "false_positives": confusion_matrix.false_positives,
            "true_negatives": confusion_matrix.true_negatives,
            "false_negatives": confusion_matrix.false_negatives,
            "total_samples": confusion_matrix.total
        },
        "metrics": {
            "accuracy": float(confusion_matrix.accuracy),
            "precision": float(confusion_matrix.precision),
            "recall": float(confusion_matrix.recall),
            "f1_score": float(confusion_matrix.f1_score),
            "false_positive_rate": float(confusion_matrix.false_positive_rate),
            "false_negative_rate": float(confusion_matrix.false_negative_rate),
            "inv_acc": float(inv_acc),
            "value_mse": float(value_mse),
        }
    }
    
    # Save as JSON
    results_path = os.path.join(test_dir, f"results_step_{checkpoint_step}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Also save a summary text file
    summary_path = os.path.join(test_dir, f"summary_step_{checkpoint_step}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Results (Step {checkpoint_step})\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives:  {confusion_matrix.true_positives:6d}\n")
        f.write(f"  False Positives: {confusion_matrix.false_positives:6d}\n")
        f.write(f"  True Negatives:  {confusion_matrix.true_negatives:6d}\n")
        f.write(f"  False Negatives: {confusion_matrix.false_negatives:6d}\n")
        f.write(f"  Total Samples:   {confusion_matrix.total:6d}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:  {confusion_matrix.accuracy:7.4f}\n")
        f.write(f"  Precision: {confusion_matrix.precision:7.4f}\n")
        f.write(f"  Recall:    {confusion_matrix.recall:7.4f}\n")
        f.write(f"  F1 Score:  {confusion_matrix.f1_score:7.4f}\n")
        f.write(f"  Invariant Accuracy: {inv_acc:7.4f}\n")
        f.write(f"  Value MSE: {value_mse:10.6f}\n\n")

        f.write("Error Rates:\n")
        f.write(f"  False Positive Rate: {confusion_matrix.false_positive_rate:7.4f}\n")
        f.write(f"  False Negative Rate: {confusion_matrix.false_negative_rate:7.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


def evaluate(
    safety_analysis_root: str,
    model_subdir: str,
    dataset_path: Optional[str] = None,
    batch_size: int = 2048,
    device: Optional[str] = None
) -> ConfusionMatrix:
    """Evaluate a trained model on test data.
    
    Args:
        safety_analysis_root: Experiment name under logs/safety_analysis/
        model_subdir: Model subdirectory name (e.g., 'dpe_lam0.9')
        dataset_path: Path to test dataset (default: logs/safety_analysis/<safety_analysis_root>/data_processed_test.pkl)
        batch_size: Batch size for evaluation
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Returns:
        ConfusionMatrix with evaluation results
    """
    # Build paths
    data_dir = os.path.join(SAFETY_ANALYSIS_BASE, safety_analysis_root)
    model_dir = os.path.join(data_dir, "results", model_subdir)
    
    # Resolve dataset path (λ-reachability still evaluates on pointwise test set for compatibility)
    if dataset_path is None:
        dataset_path = os.path.join(data_dir, "data_processed_test.pkl")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {dataset_path}")

    models_dir = os.path.join(model_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")

    # Load training config to discover algorithm and hidden dimensions
    config_path = os.path.join(model_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, 'r') as f:
        training_config = json.load(f)

    algorithm = training_config.get("algorithm", "dpe")
    hidden_dims = tuple(training_config.get("hidden_dims", [256, 256]))

    # Find latest checkpoint
    step, checkpoint_path = find_latest_checkpoint(models_dir)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Training step: {step}")
    print(f"Algorithm: {algorithm}")

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    print(f"Device: {torch_device}")

    # Load dataset
    dataset = HJValueDataset(dataset_path)
    input_dim = dataset.x.shape[1]
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(dataset)}, Input dimension: {input_dim}")

    print(f"Model architecture: input_dim={input_dim}, hidden_dims={hidden_dims}")

    # Load model
    model = load_model(checkpoint_path, input_dim, hidden_dims, torch_device, algorithm)
    print("Model loaded successfully")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=hj_collate)

    # Evaluate
    print("\nEvaluating...")
    confusion_matrix, inv_acc, value_mse = compute_confusion_matrix(model, dataloader, torch_device)

    # Save results
    save_results(confusion_matrix, model_dir, step, inv_acc, value_mse)

    return confusion_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained safety analysis model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--safety_analysis_root',
        type=str,
        required=True,
        help='Safety analysis experiment name (e.g., "g1_flat_ppo_1000")'
    )
    parser.add_argument(
        '--model_subdir',
        type=str,
        required=True,
        help='Model subdirectory name (e.g., "dpe_lam0.9")'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to test dataset (default: logs/safety_analysis/<safety_analysis_root>/data_processed_test.pkl)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for evaluation (default: auto-detect)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Run evaluation
    confusion_matrix = evaluate(
        safety_analysis_root=args.safety_analysis_root,
        model_subdir=args.model_subdir,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Print results
    confusion_matrix.print_summary()


if __name__ == "__main__":
    main()
