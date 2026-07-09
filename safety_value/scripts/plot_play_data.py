#!/usr/bin/env python3
"""
Plot step-wise signals from safety analysis inference data.

This script plots safety signals, event flags, and safety value predictions
from the inference dataset, supporting the new data structure with flexible
recorder naming.

Usage:
    python plot_play_data.py --data_dir /path/to/seed_42
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Optional


def find_signal_keys(episode_group: h5py.Group) -> Dict[str, str]:
    """
    Find signal keys in episode data using pattern matching.
    
    Args:
        episode_group: HDF5 group containing episode data
        
    Returns:
        Dictionary mapping signal types to their actual keys
    """
    keys = list(episode_group.keys())
    signals = {}
    
    # Find safety_signal_* (exactly 1 expected)
    safety_keys = [k for k in keys if k.startswith("safety_signal_")]
    if len(safety_keys) > 0:
        signals["safety_signal"] = safety_keys[0]
    
    # Find event_* (exactly 1 expected)
    event_keys = [k for k in keys if k.startswith("event_")]
    if len(event_keys) > 0:
        signals["event"] = event_keys[0]
    
    # Find terminal_state (exact match)
    if "terminal_state" in keys:
        signals["terminal_state"] = "terminal_state"
    
    # Find safety_value (exact match or pattern)
    if "safety_value" in keys:
        signals["safety_value"] = "safety_value"
    else:
        safety_value_keys = [k for k in keys if "safety_value" in k.lower()]
        if len(safety_value_keys) > 0:
            signals["safety_value"] = safety_value_keys[0]
    
    return signals


def plot_episode_data(data_dir: str, episode_name: str = None, output_dir: str = None):
    """
    Load and plot safety signals for an episode from inference data.
    
    Args:
        data_dir: Directory containing dataset.hdf5
        episode_name: Name of episode to plot (e.g., "demo_0"). If None, plots the first episode.
        output_dir: Directory to save plots. If None, uses data_dir/plots/
    """
    # Construct path to dataset
    dataset_path = os.path.join(data_dir, "dataset.hdf5")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Open HDF5 file
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        
        # Get episode names
        episode_names = list(data_group.keys())
        print(f"Found {len(episode_names)} episodes: {episode_names}")
        
        # Select episode
        if episode_name is None:
            episode_name = episode_names[0]
            print(f"No episode specified, using first episode: {episode_name}")
        elif episode_name not in episode_names:
            raise ValueError(f"Episode {episode_name} not found. Available: {episode_names}")
        
        # Load episode data
        episode_group = data_group[episode_name]
        
        # Check what data is available
        print(f"\nData available in episode {episode_name}:")
        available_keys = list(episode_group.keys())
        print(f"  Available keys: {available_keys}")
        
        # Find signal keys using pattern matching
        signal_keys = find_signal_keys(episode_group)
        print(f"  Discovered signals: {list(signal_keys.keys())}")
        
        # Load data based on discovered keys
        safety_signal_key = signal_keys.get("safety_signal")
        event_key = signal_keys.get("event")
        safety_value_key = signal_keys.get("safety_value")
        
        if safety_signal_key is None:
            print("  Warning: No safety_signal found")
            l_values = None
        else:
            l_values = np.array(episode_group[safety_signal_key])
            print(f"  {safety_signal_key} shape: {l_values.shape}")
        
        if event_key is None:
            print("  Warning: No event signal found")
            push_flag = None
        else:
            push_flag = np.array(episode_group[event_key])
            print(f"  {event_key} shape: {push_flag.shape}")
        
        if safety_value_key is None:
            print("  Warning: No safety_value found")
            safety_values = None
        else:
            safety_values = np.array(episode_group[safety_value_key])
            print(f"  {safety_value_key} shape: {safety_values.shape}")
        
        # Check if we have the required data
        if l_values is None:
            raise ValueError("Required data (safety_signal) not found in episode")
        
        # Create plots
        num_steps = len(l_values)
        steps = np.arange(num_steps)
        
        # Determine which plots to show
        has_event = push_flag is not None
        has_safety_value = safety_values is not None
        
        num_plots = 1  # Always have safety signal
        if has_event:
            num_plots += 1
        if has_safety_value:
            num_plots += 1
        
        # Smaller figure with larger fonts for denser information
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 2.5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Event flag (if available)
        if has_event:
            ax = axes[plot_idx]
            plot_idx += 1
            ax.plot(steps, push_flag, 'o-', color='red', markersize=3, label='event_flag')
            ax.set_ylabel('Event Flag', fontsize=13, fontweight='bold')
            ax.set_title(f'Episode: {episode_name}', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax.set_ylim(-0.1, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Highlight push events
            push_indices = np.where(push_flag > 0.5)[0]
            for idx in push_indices:
                ax.axvline(x=idx, color='red', alpha=0.3, linestyle='--', linewidth=1)
        else:
            push_indices = []
        
        # Plot 2: Safety signal (l)
        ax = axes[plot_idx]
        plot_idx += 1
        ax.plot(steps, l_values, '-', color='blue', linewidth=2.5, label='safety_signal')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='l=0 (safe threshold)')
        ax.set_ylabel('Safety Signal', fontsize=13, fontweight='bold')
        if not has_event:
            ax.set_title(f'Episode: {episode_name}', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_ylim(-1.5, 3.0)
        
        # Highlight push events
        if has_event:
            for idx in push_indices:
                ax.axvline(x=idx, color='red', alpha=0.3, linestyle='--', linewidth=1)
        
        # Shade safe region (l <= 0)
        ax.fill_between(steps, l_values, 0, where=(l_values <= 0), 
                        color='green', alpha=0.2, label='Safe region')
        ax.fill_between(steps, l_values, 0, where=(l_values > 0), 
                        color='red', alpha=0.2, label='Unsafe region')
        
        # Plot 3: Safety value prediction (if available)
        if has_safety_value:
            ax = axes[plot_idx]
            plot_idx += 1
            ax.plot(steps, safety_values, '-', color='purple', linewidth=2.5, label='safety_value (predicted)')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='value=0 (safe threshold)')
            ax.set_ylabel('Safety Value (Predicted)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.set_ylim(-1.5, 3.0)
            
            # Highlight push events
            if has_event:
                for idx in push_indices:
                    ax.axvline(x=idx, color='red', alpha=0.3, linestyle='--', linewidth=1)
            
            # Shade safe region (value <= 0)
            ax.fill_between(steps, safety_values, 0, where=(safety_values <= 0), 
                            color='green', alpha=0.2, label='Safe region')
            ax.fill_between(steps, safety_values, 0, where=(safety_values > 0), 
                            color='red', alpha=0.2, label='Unsafe region')
        else:
            axes[-1].set_xlabel('Time Step', fontsize=13, fontweight='bold')
            axes[-1].tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        
        # Save plot
        if output_dir is None:
            output_dir = os.path.join(data_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{episode_name}_plot.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total steps: {num_steps}")
        
        if has_event:
            print(f"Number of events: {len(push_indices)}")
            if len(push_indices) > 0:
                print(f"Event indices: {push_indices.tolist()}")
        
        print(f"\nSafety signal:")
        print(f"  Min: {l_values.min():.4f}")
        print(f"  Max: {l_values.max():.4f}")
        print(f"  Mean: {l_values.mean():.4f}")
        print(f"  Safe steps (l <= 0): {np.sum(l_values <= 0)} ({100*np.sum(l_values <= 0)/num_steps:.1f}%)")
        
        if has_safety_value:
            print(f"\nSafety Value (predicted):")
            print(f"  Min: {safety_values.min():.4f}")
            print(f"  Max: {safety_values.max():.4f}")
            print(f"  Mean: {safety_values.mean():.4f}")
            print(f"  Safe steps (value <= 0): {np.sum(safety_values <= 0)} ({100*np.sum(safety_values <= 0)/num_steps:.1f}%)")
            
            # Correlation analysis
            correlation = np.corrcoef(l_values, safety_values)[0, 1]
            print(f"  Correlation with safety_signal: {correlation:.4f}")
        
        print("="*60)
        
        return fig, axes


def plot_all_episodes(data_dir: str, output_dir: Optional[str] = None):
    """
    Plot all episodes in the dataset file.
    
    Args:
        data_dir: Directory containing dataset.hdf5
        output_dir: Directory to save plots. If None, uses data_dir/plots/
    """
    dataset_path = os.path.join(data_dir, "dataset.hdf5")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        episode_names = list(data_group.keys())
    
    print(f"Plotting {len(episode_names)} episodes...")
    
    for episode_name in episode_names:
        try:
            plot_episode_data(data_dir, episode_name, output_dir)
            plt.close('all')  # Close to free memory
        except Exception as e:
            print(f"Error plotting {episode_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nFinished plotting all episodes!")


def plot_seed_directory(seed_dir: str):
    """
    Plot all episodes from a seed directory.
    
    This is a convenience function that takes a seed directory path
    (e.g., path/to/seed_42) and plots all episodes in the dataset.hdf5
    file, saving plots to seed_dir/plots/.
    
    Args:
        seed_dir: Path to seed directory containing dataset.hdf5
    """
    seed_dir = os.path.abspath(os.path.expanduser(seed_dir))
    
    if not os.path.exists(seed_dir):
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")
    
    dataset_path = os.path.join(seed_dir, "dataset.hdf5")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print("="*80)
    print(f"Plotting all episodes from: {seed_dir}")
    print("="*80)
    
    # Set output directory to seed_dir/plots
    output_dir = os.path.join(seed_dir, "plots")
    
    # Plot all episodes
    plot_all_episodes(seed_dir, output_dir)
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot step-wise signals from safety analysis inference data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing dataset.hdf5 (e.g., path/to/seed_42)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Name of episode to plot (e.g., 'demo_0'). If not specified, plots the first episode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all episodes in the file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots. If not specified, uses data_dir/plots/",
    )
    
    args = parser.parse_args()
    
    # Expand path
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if args.all:
        plot_all_episodes(data_dir, args.output_dir)
    else:
        plot_episode_data(data_dir, args.episode, args.output_dir)
        plt.show()


if __name__ == "__main__":
    # Hardcoded path for quick testing - modify this to your seed directory
    seed_dir = "logs/safety_analysis/g1_flat_ppo_1000/dpe/inference/seed_42"
    # seed_dir = "logs/safety_analysis/g1_flat_ppo_1000/supervised/inference/seed_42"
    
    # Check if the hardcoded path exists
    if os.path.exists(seed_dir):
        print("Using hardcoded seed directory for plotting...")
        plot_seed_directory(seed_dir)
    else:
        print(f"Hardcoded seed directory not found: {seed_dir}")
        print("Falling back to command-line argument parsing...")
        main()
