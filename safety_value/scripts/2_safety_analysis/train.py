"""
Training script for safety analysis algorithms.

This script trains safety value functions using processed trajectory data.
Supports multiple algorithms:
- dpe: Temporal difference learning with discount factor
- weakly_supervised: Direct weakly supervised learning with margin-based loss
- supervised: Supervised regression to directly fit sample safety values

Usage:
    python safety_value/scripts/safety_analysis/train.py \\
        --safety_analysis_root exp_name \\
        --algo dpe \\
        --epochs 20 \\
        --batch 256
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_value.safety_analysis_algos.dataset import HJValueDataset, hj_collate
from safety_value.safety_analysis_algos.dpe import DiscountedPolicyEvaluationTrainer
from safety_value.safety_analysis_algos.weakly_supervised import WeaklySupervisedTrainer
from safety_value.safety_analysis_algos.supervised import SupervisedTrainer
from safety_value.safety_analysis_algos.lambda_reachability import LambdaReachabilityTrainer


# Base directory for all safety analysis data and results
SAFETY_ANALYSIS_BASE = "logs/safety_analysis"


# Algorithm configurations
ALGORITHMS = {
    'dpe': {
        'trainer': DiscountedPolicyEvaluationTrainer,
    },
    'weakly_supervised': {
        'trainer': WeaklySupervisedTrainer,
    },
    'supervised': {
        'trainer': SupervisedTrainer,
    },
    'lambda_reachability': {
        'trainer': LambdaReachabilityTrainer,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train safety analysis models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--safety_analysis_root',
        type=str,
        required=True,
        help='Safety analysis experiment name (data will be read from and saved to logs/safety_analysis/<safety_analysis_root>)'
    )
    data_group.add_argument(
        '--train_pkl',
        type=str,
        default=None,
        help='Path to train pkl (default: logs/safety_analysis/<safety_analysis_root>/data_processed_train.pkl)'
    )
    data_group.add_argument(
        '--test_pkl',
        type=str,
        default=None,
        help='Path to test pkl (default: logs/safety_analysis/<safety_analysis_root>/data_processed_test.pkl)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    train_group.add_argument('--batch', type=int, default=256, help='Batch size')
    train_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_group.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256], help='Hidden layer dimensions')
    train_group.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    train_group.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluate on train/test every N steps'
    )
    train_group.add_argument(
        '--eval_batches',
        type=int,
        default=0,
        help='Number of samples to use for each evaluation (0 means use the full evaluation loader)'
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for training reproducibility'
    )
    train_group.add_argument(
        '--total_steps',
        type=int,
        default=None,
        help='Total training steps (overrides epoch count when set)'
    )
    train_group.add_argument(
        '--save_all_checkpoints',
        action='store_true',
        help='Save checkpoints throughout training (default: only save the final model as last.pt)'
    )
    train_group.add_argument(
        '--results_subdir',
        type=str,
        default='',
        help='Optional subdirectory under logs/safety_analysis/<root>/results to store this run.'
    )
    
    # Algorithm arguments
    algo_group = parser.add_argument_group('Algorithm')
    algo_group.add_argument(
        '--algo',
        type=str,
        default='dpe',
        choices=list(ALGORITHMS.keys()),
        help='Safety analysis algorithm to use'
    )
    algo_group.add_argument(
        '--run_name',
        type=str,
        default='',
        help='Custom run name to append to output directory'
    )
    algo_group.add_argument(
        '--model_tag',
        type=str,
        default=None,
        help='Optional tag to use as the base directory name (defaults to algo)'
    )
    
    # Weakly supervised algorithm arguments
    weakly_supervised_group = parser.add_argument_group('Weakly supervised algorithm')
    weakly_supervised_group.add_argument(
        '--margin',
        type=float,
        default=0.1,
        help='Margin for weakly supervised loss'
    )
    
    # Discounted policy evaluation arguments
    dpe_group = parser.add_argument_group('Discounted policy evaluation')
    dpe_group.add_argument(
        '--lam',
        type=float,
        default=0.0,
        help='Initial discount factor (lambda)'
    )
    dpe_group.add_argument(
        '--disable_lambda_annealing',
        dest='enable_lambda_annealing',
        action='store_false',
        default=True,
        help='Disable lambda annealing (enabled by default)'
    )
    dpe_group.add_argument(
        '--lambda_update_steps',
        type=int,
        default=500,
        help='Update lambda every N steps'
    )
    dpe_group.add_argument(
        '--lambda_max',
        type=float,
        default=0.99,
        help='Maximum lambda value for annealing'
    )
    dpe_group.add_argument(
        '--lambda_increase_ratio',
        type=float,
        default=0.1,
        help='Fraction of remaining gap to lambda_max to add when annealing triggers'
    )
    dpe_group.add_argument(
        '--loss_converge_tol',
        type=float,
        default=0.01,
        help='Relative loss-change threshold for triggering lambda annealing'
    )
    target_group = dpe_group.add_mutually_exclusive_group()
    target_group.add_argument(
        '--use_target_network',
        dest='use_target_network',
        action='store_true',
        help='Enable target network (default: enabled, similar to DQN)'
    )
    target_group.add_argument(
        '--no_target_network',
        dest='use_target_network',
        action='store_false',
        help='Disable target network (not recommended)'
    )
    dpe_group.add_argument(
        '--target_update_steps',
        type=int,
        default=10,
        help='Update target network every N steps'
    )
    dpe_group.add_argument(
        '--target_tau',
        type=float,
        default=0.05,
        help='Soft update parameter for target network'
    )

    # Lambda reachability arguments
    lambda_group = parser.add_argument_group('Lambda reachability')
    lambda_group.add_argument('--lambda_param', type=float, default=0.5, help='Geometric horizon parameter λ')
    lambda_group.add_argument('--max_horizon', type=int, default=200, help='Maximum sampled horizon')
    lambda_group.add_argument('--alpha_bce', type=float, default=5.0, help='Scale for BCE logits')
    lambda_group.add_argument('--weight_main', type=float, default=1.0, help='Weight for main regression loss')
    lambda_group.add_argument('--weight_hinge_lb', type=float, default=1.0, help='Weight for lower-bound hinge loss')
    lambda_group.add_argument('--weight_hinge_mono', type=float, default=1.0, help='Weight for monotonicity hinge loss')
    lambda_group.add_argument('--weight_bce', type=float, default=1.0, help='Weight for BCE sign loss')
    lambda_group.add_argument('--target_tau_lambda', type=float, default=0.005, help='Target Polyak tau')
    lambda_group.add_argument('--target_update_period', type=int, default=1, help='Target update period (steps)')
    
    parser.set_defaults(use_target_network=True)
    return parser.parse_args()


def resolve_data_paths(args: argparse.Namespace) -> Tuple[str, str, Optional[str]]:
    """Resolve data directory and dataset paths per algorithm."""
    data_dir = os.path.join(SAFETY_ANALYSIS_BASE, args.safety_analysis_root)

    train_pkl_path = args.train_pkl or os.path.join(data_dir, 'data_processed_train.pkl')
    test_pkl_path = args.test_pkl or os.path.join(data_dir, 'data_processed_test.pkl')

    if not os.path.exists(train_pkl_path):
        raise FileNotFoundError(f"Training data not found: {train_pkl_path}")

    if not os.path.exists(test_pkl_path):
        print(f"Warning: Test data not found at {test_pkl_path}, training without validation")
        test_pkl_path = None

    return data_dir, train_pkl_path, test_pkl_path


def build_output_directory(args: argparse.Namespace, data_dir: str) -> Tuple[str, str, str, str]:
    """Build output directory structure, including per-seed train artifacts.

    Returns (output_dir, checkpoint_dir, train_data_dir, plot_dir) where plot_dir is
    the seed-specific plots directory under train/seed_<seed>/plots and train_data_dir
    is train/seed_<seed>.
    """
    run_name = args.run_name or ''
    base_tag = args.model_tag or args.algo
    model_subdir = f"{base_tag}_{run_name}" if run_name else base_tag

    results_root = os.path.join(data_dir, 'results')
    if getattr(args, 'results_subdir', ''):
        results_root = os.path.join(results_root, args.results_subdir)

    output_dir = os.path.join(results_root, model_subdir)
    checkpoint_dir = os.path.join(output_dir, 'models')

    train_root = os.path.join(output_dir, 'train')
    seed_dir = os.path.join(train_root, f"seed_{args.seed}")
    plot_dir = os.path.join(seed_dir, 'plots')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return output_dir, checkpoint_dir, seed_dir, plot_dir


def save_training_config(args: argparse.Namespace, output_dir: str, input_dim: int):
    """Save training configuration to JSON file.
    
    Args:
        args: Parsed command line arguments
        output_dir: Output directory for this training run
        input_dim: Input dimension of the model
    """
    run_name = args.run_name or ''

    config = {
        'algorithm': args.algo,
        'model_tag': args.model_tag or args.algo,
        'safety_analysis_root': args.safety_analysis_root,
        'input_dim': input_dim,
        
        # Training hyperparameters
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'hidden_dims': args.hidden_dims,
        'device': args.device,
        'eval_steps': args.eval_steps,
        'eval_batches': args.eval_batches,
        'total_steps': args.total_steps,
        'run_name': run_name,
        'results_subdir': getattr(args, 'results_subdir', ''),
    }
    
    # Add algorithm-specific parameters
    if args.algo == 'dpe':
        config['dpe'] = {
            'initial_lambda': args.lam,
            'enable_lambda_annealing': args.enable_lambda_annealing,
            'lambda_update_steps': args.lambda_update_steps,
            'lambda_max': args.lambda_max,
            'lambda_increase_ratio': args.lambda_increase_ratio,
            'loss_converge_tol': args.loss_converge_tol,
            'use_target_network': args.use_target_network,
            'target_update_steps': args.target_update_steps,
            'target_tau': args.target_tau,
        }
    elif args.algo == 'weakly_supervised':
        config['weakly_supervised'] = {
            'margin': args.margin,
        }
    elif args.algo == 'lambda_reachability':
        config['lambda_reachability'] = {
            'lambda_param': args.lambda_param,
            'max_horizon': args.max_horizon,
            'alpha_bce': args.alpha_bce,
            'weight_main': args.weight_main,
            'weight_hinge_lb': args.weight_hinge_lb,
            'weight_hinge_mono': args.weight_hinge_mono,
            'weight_bce': args.weight_bce,
            'target_tau': args.target_tau_lambda,
            'target_update_period': args.target_update_period,
        }
    
    # Save to JSON file
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved training configuration to {config_path}")


def create_trainer(args: argparse.Namespace, input_dim: int, train_ds, test_ds, plot_dir: str, train_data_dir: str):
    """Create trainer instance based on algorithm selection.
    
    Args:
        args: Parsed command line arguments
        input_dim: Input dimension
        train_ds: Training dataset
        test_ds: Test dataset (may be None)
        plot_dir: Directory for plots
        train_data_dir: Directory for training data
        
    Returns:
        Trainer instance
    """
    trainer_cls = ALGORITHMS[args.algo]['trainer']
    
    common_kwargs = {
        'input_dim': input_dim,
        'train_dataset': train_ds,
        'test_dataset': test_ds,
        'device': args.device,
        'batch_size': args.batch,
        'lr': args.lr,
        'eval_steps': args.eval_steps,
        'eval_batches': args.eval_batches,
        'plot_dir': plot_dir,
        'train_data_dir': train_data_dir,
        'hidden_dims': tuple(args.hidden_dims),
        'collate_fn': hj_collate,
    }
    
    if args.algo == 'dpe':
        trainer = trainer_cls(
            **common_kwargs,
            initial_lambda          = args.lam,
            lambda_update_steps     = args.lambda_update_steps,
            lambda_max              = args.lambda_max,
            lambda_increase_ratio   = args.lambda_increase_ratio,
            loss_converge_tol       = args.loss_converge_tol,
            use_target_network      = args.use_target_network,
            target_update_steps     = args.target_update_steps,
            target_tau              = args.target_tau,
            enable_lambda_annealing = args.enable_lambda_annealing,
        )
    elif args.algo == 'weakly_supervised':
        trainer = trainer_cls(
            **common_kwargs,
            margin = args.margin,
        )
    elif args.algo == 'supervised':
        trainer = trainer_cls(
            **common_kwargs,
        )
    elif args.algo == 'lambda_reachability':
        trainer = trainer_cls(
            **common_kwargs,
            lambda_param=args.lambda_param,
            max_horizon=args.max_horizon,
            alpha_bce=args.alpha_bce,
            weight_main=args.weight_main,
            weight_hinge_lb=args.weight_hinge_lb,
            weight_hinge_mono=args.weight_hinge_mono,
            weight_bce=args.weight_bce,
            target_tau=args.target_tau_lambda,
            target_update_period=args.target_update_period,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    return trainer


def print_config(args: argparse.Namespace, output_dir: str):
    """Print training configuration."""
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Total steps       : {args.total_steps or (args.epochs * args.eval_steps)}")
    
    if args.algo == 'dpe':
        print(f"\n  Discounted Policy Evaluation Settings:")
        print(f"    Lambda (initial): {args.lam}")
        print(f"    Lambda annealing: {'Enabled' if args.enable_lambda_annealing else 'Disabled'}")
        if args.enable_lambda_annealing:
            print(f"      - Update steps: {args.lambda_update_steps}")
            print(f"      - Lambda max: {args.lambda_max}")
            print(f"      - Increase ratio: {args.lambda_increase_ratio}")
            print(f"      - Loss converge tol (relative): {args.loss_converge_tol}")
        print(f"    Target network: {'Enabled' if args.use_target_network else 'Disabled'}")
        if args.use_target_network:
            print(f"      - Update steps: {args.target_update_steps}")
            print(f"      - Tau: {args.target_tau}")
    elif args.algo == 'weakly_supervised':
        print(f"\n  Weakly Supervised Learning Settings:")
        print(f"    Margin: {args.margin}")
    elif args.algo == 'supervised':
        print(f"\n  Supervised Regression Settings:")
        print(f"    Loss: Mean Squared Error (MSE)")
    elif args.algo == 'lambda_reachability':
        print(f"\n  Lambda Reachability Settings:")
        print(f"    lambda_param: {args.lambda_param}")
        print(f"    max_horizon: {args.max_horizon}")
        print(f"    alpha_bce: {args.alpha_bce}")
        print(f"    weight_main: {args.weight_main}")
        print(f"    weight_hinge_lb: {args.weight_hinge_lb}")
        print(f"    weight_hinge_mono: {args.weight_hinge_mono}")
        print(f"    weight_bce: {args.weight_bce}")
        print(f"    target_tau: {args.target_tau_lambda}")
        print(f"    target_update_period: {args.target_update_period}")
    
    print("=" * 80 + "\n")


def evaluate_final_accuracy(trainer, dataset, batch_size: int, split_name: str, algo: str):
    """Evaluate final accuracy on a dataset split."""
    # Always use hj_collate to keep variable-length futures as lists and avoid stacking errors
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=hj_collate)

    stats = trainer.evaluate_invariant_sign(dataloader)

    # lambda trainer returns acc directly; others return preds/labels
    if 'acc' in stats and not np.isnan(stats['acc']):
        print(f'Final {split_name} invariant sign accuracy: {float(stats["acc"]):.4f}')
        return

    if len(stats.get("labels", [])) > 0:
        preds = stats["preds"]
        labels = stats["labels"]
        accuracy = float((preds.ravel() == labels.ravel()).mean())
        print(f'Final {split_name} invariant sign accuracy: {accuracy:.4f}')


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Resolve data paths
    data_dir, train_pkl_path, test_pkl_path = resolve_data_paths(args)
    
    print(f"Data directory: {data_dir}")
    
    # Load datasets (unified format for all algos)
    train_ds = HJValueDataset(train_pkl_path)
    test_ds = HJValueDataset(test_pkl_path) if test_pkl_path is not None else None
    input_dim = train_ds.x.shape[1]
    print(f"Loaded training data: {len(train_ds)} samples, input_dim={input_dim}")
    if test_ds is not None:
        print(f"Loaded test data: {len(test_ds)} samples")
    
    # Build output directory (per-run) with seed-specific metrics/plots handled inside helper
    output_dir, checkpoint_dir, train_data_dir, plot_dir = build_output_directory(args, data_dir)
    
    # Save training configuration
    save_training_config(args, output_dir, input_dim)
    
    # Print configuration
    print_config(args, output_dir)
    
    # Create trainer
    trainer = create_trainer(
        args=args,
        input_dim=input_dim,
        train_ds=train_ds,
        test_ds=test_ds,
        plot_dir=plot_dir,
        train_data_dir=train_data_dir,
    )
    
    # Train
    print("Starting training...")
    total_steps = args.total_steps
    if total_steps is None:
        # Fallback: approximate from epochs * eval_steps if total_steps not provided
        total_steps = int(args.epochs * args.eval_steps)
        print(f"[INFO] total_steps not provided; using epochs*eval_steps = {total_steps}")

    run_suffix = f"_seed_{args.seed}"

    # Train model
    trainer.train(
        total_steps=total_steps,
        # Always provide the checkpoint directory so the final model is saved as last*.pt.
        # The --save_all_checkpoints flag now only controls whether intermediate checkpoints
        # are stored during evaluation steps.
        ckpt_dir=checkpoint_dir,
        log_every=100,
        save_last_only=not args.save_all_checkpoints,
        run_suffix=run_suffix,
    )
    print("\nTraining complete!")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    evaluate_final_accuracy(trainer, train_ds, args.batch, 'train', args.algo)
    if test_ds is not None:
        evaluate_final_accuracy(trainer, test_ds, args.batch, 'test', args.algo)


if __name__ == '__main__':
    main()
