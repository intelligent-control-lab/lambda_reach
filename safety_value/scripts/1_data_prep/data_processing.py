"""
Process safety analysis dataset with flexible signal matching.

This script processes rollout data for HJ value training, supporting flexible
recorder naming conventions through pattern matching.
"""

import h5py
import os
import numpy as np
import pickle
from typing import Dict, Tuple, List


def find_signal_keys(episode_group: h5py.Group) -> Dict[str, str]:
    """
    Find signal keys in episode data using pattern matching.
    
    Args:
        episode_group: HDF5 group containing episode data
        
    Returns:
        Dictionary mapping signal types to their actual keys
        
    Raises:
        ValueError: If required signals are missing or if there are multiple matches
    """
    keys = list(episode_group.keys())
    signals = {}
    
    # 1. Find safety_signal_* (exactly 1 expected)
    safety_keys = [k for k in keys if k.startswith("safety_signal_")]
    if len(safety_keys) == 0:
        raise ValueError(f"No safety_signal_* key found. Available keys: {keys}")
    if len(safety_keys) > 1:
        raise ValueError(f"Multiple safety_signal_* keys found: {safety_keys}. Expected exactly 1.")
    signals["safety_signal"] = safety_keys[0]
    
    # 2. Find event_* (exactly 1 expected)
    event_keys = [k for k in keys if k.startswith("event_")]
    if len(event_keys) == 0:
        raise ValueError(f"No event_* key found. Available keys: {keys}")
    if len(event_keys) > 1:
        raise ValueError(f"Multiple event_* keys found: {event_keys}. Expected exactly 1.")
    signals["event"] = event_keys[0]
    
    # 3. Find terminal_state (exact match required)
    if "terminal_state" not in keys:
        raise ValueError(f"Required key 'terminal_state' not found. Available keys: {keys}")
    signals["terminal_state"] = "terminal_state"
    
    # 4. Find policy_obs (exact match required)
    if "policy_obs" not in keys:
        raise ValueError(f"Required key 'policy_obs' not found. Available keys: {keys}")
    signals["policy_obs"] = "policy_obs"
    
    # 5. Find stability_signal_* (optional - only used for visualization, not training)
    stability_keys = [k for k in keys if k.startswith("stability_signal_")]
    if len(stability_keys) == 0:
        # Stability signal is optional - set to None to indicate it's not present
        signals["stability_signal"] = None
    elif len(stability_keys) > 1:
        raise ValueError(f"Multiple stability_signal_* keys found: {stability_keys}. Expected at most 1.")
    else:
        signals["stability_signal"] = stability_keys[0]
    
    return signals


def load_episode_data(episode_group: h5py.Group, signal_keys: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Load episode data using the discovered signal keys.
    
    Args:
        episode_group: HDF5 group containing episode data
        signal_keys: Dictionary mapping signal types to their actual keys
        
    Returns:
        Dictionary containing loaded numpy arrays
    """
    result = {
        "safety_signal": np.asarray(episode_group[signal_keys["safety_signal"]]),
        "event_flag": np.asarray(episode_group[signal_keys["event"]]),
        "terminal_flag": np.asarray(episode_group[signal_keys["terminal_state"]]),
        "policy_obs": np.asarray(episode_group[signal_keys["policy_obs"]]),
    }
    
    # Stability flag is optional - create zeros if not present
    if signal_keys.get("stability_signal") is not None:
        result["stability_flag"] = np.asarray(episode_group[signal_keys["stability_signal"]])
    else:
        # Create dummy zeros array matching safety_signal length
        result["stability_flag"] = np.zeros_like(result["safety_signal"])
    
    return result


def print_data_composition(raw_dataset_path: str) -> None:
    """
    Print the composition of the dataset for sanity checking.
    
    Args:
        raw_dataset_path: Path to the raw HDF5 dataset
    """
    print("="*80)
    print("DATA COMPOSITION REPORT")
    print("="*80)
    
    with h5py.File(raw_dataset_path, "r") as f:
        raw_data = f["data"]
        n_episodes = len(raw_data.keys())
        print(f"\nTotal episodes: {n_episodes}")
        
        # Check first episode for signal keys
        first_episode_key = list(raw_data.keys())[0]
        first_episode = raw_data[first_episode_key]
        
        try:
            signal_keys = find_signal_keys(first_episode)
            print("\nDiscovered signal mappings:")
            for signal_type, actual_key in signal_keys.items():
                print(f"  {signal_type:20s} -> {actual_key}")
            
            # Load and report statistics
            data = load_episode_data(first_episode, signal_keys)
            print(f"\nSample episode '{first_episode_key}' statistics:")
            print(f"  Episode length: {len(data['safety_signal'])} steps")
            print(f"  Safety signal shape: {data['safety_signal'].shape}")
            print(f"  Event occurrences: {int(data['event_flag'].sum())}")
            print(f"  Stability flag sum: {int(data['stability_flag'].sum())} {'(dummy zeros - not recorded)' if signal_keys.get('stability_signal') is None else ''}")
            print(f"  Terminal flag sum: {int(data['terminal_flag'].sum())}")
            print(f"  Policy obs shape: {data['policy_obs'].shape}")
            
            print("\n" + "="*80)
            print("SANITY CHECK PASSED - Proceeding with data processing...")
            print("="*80 + "\n")
            
        except ValueError as e:
            print(f"\n[ERROR] Sanity check failed: {e}")
            raise


def segment_by_events(
    data: Dict[str, np.ndarray],
    use_events: bool = True,
) -> Tuple[List[np.ndarray], ...]:
    """
    Optionally segment episode data by event occurrences.

    If ``use_events`` is False, the entire episode is treated as a single segment.
    When True, we split at event boundaries but keep all resulting segments.
    """
    if not use_events:
        return (
            [data["safety_signal"]],
            [data["stability_flag"]],
            [data["policy_obs"]],
        )

    event_indices = np.where(data["event_flag"] > 0)[0]

    safety_signal_segs = np.split(data["safety_signal"], event_indices)
    stability_segs = np.split(data["stability_flag"], event_indices)
    obs_segs = np.split(data["policy_obs"], event_indices)
    return safety_signal_segs, stability_segs, obs_segs


def future_max(signal: np.ndarray) -> np.ndarray:
    """Compute max over all future steps (unbounded lookahead)."""
    return np.maximum.accumulate(signal[::-1])[::-1]


def process_segment(
    safety_signal_seg: np.ndarray,
    _stab_seg: np.ndarray,
    obs_seg: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
]:
    """
    Process a single segment to extract training data.

    Legacy outputs remain (x, x_next, etc.). λ-reachability will consume
    trajectories directly in the caller. Future-dependent values use an
    unbounded lookahead over the remaining segment.
    """
    if len(obs_seg) < 2:
        # Too short to form (x, x_next)
        empty = np.empty((0,))
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
            [],
        )

    x = obs_seg[:-1]
    x_next = obs_seg[1:]
    safety_signal = safety_signal_seg[:-1]

    # Forward-looking maximum safety signal for each timestep (inclusive).
    sample_safety_value = future_max(safety_signal_seg)[:-1]

    # A state is invariant if the future safety signal never becomes positive.
    invariant_flag = (sample_safety_value <= 0).astype(np.float32)

    # Future trajectories for each state (x[t+1:]) and signals (l[t+1:]).
    x_future_traj = [obs_seg[i + 1 :] for i in range(len(x))]
    l_future_traj = [safety_signal_seg[i + 1 :] for i in range(len(x))]

    return (
        x,
        x_next,
        safety_signal,
        sample_safety_value,
        invariant_flag,
        x_future_traj,
        l_future_traj,
    )


def balance_invariant_samples(
    x: np.ndarray,
    x_next: np.ndarray,
    safety_signal: np.ndarray,
    sample_safety_value: np.ndarray,
    invariant_flag: np.ndarray,
    x_future: List[np.ndarray] | None = None,
    safety_future: List[np.ndarray] | None = None,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray] | None, List[np.ndarray] | None]:
    """
    Balance dataset by downsampling the larger class (invariant vs not invariant).
    
    Args:
        x: Current state observations
        x_next: Next state observations
        safety_signal: Safety signal values
        sample_safety_value: Maximum safety signal from each timestep to segment end
        invariant_flag: Binary invariant labels
        
    Returns:
        Tuple of balanced arrays (x, x_next, safety_signal, sample_safety_value, invariant_flag)
        If ``return_indices`` is True, also returns the chosen indices for reuse.
    """
    inv_indices = np.where(invariant_flag > 0)[0]
    not_inv_indices = np.where(invariant_flag == 0)[0]
    n_inv = len(inv_indices)
    n_not_inv = len(not_inv_indices)
    
    if n_inv == 0 or n_not_inv == 0:
        keep_indices = np.arange(len(invariant_flag))
        xf_bal = x_future if x_future is not None else None
        sf_bal = safety_future if safety_future is not None else None
        if return_indices:
            return x, x_next, safety_signal, sample_safety_value, invariant_flag, keep_indices, xf_bal, sf_bal
        return x, x_next, safety_signal, sample_safety_value, invariant_flag, xf_bal, sf_bal
    
    # Downsample the larger group
    if n_inv > n_not_inv:
        keep_inv = np.random.choice(inv_indices, n_not_inv, replace=False)
        keep_not_inv = not_inv_indices
    else:
        keep_inv = inv_indices
        keep_not_inv = np.random.choice(not_inv_indices, n_inv, replace=False)
    
    keep_indices = np.concatenate([keep_inv, keep_not_inv])
    np.random.shuffle(keep_indices)
    
    xf_bal = None if x_future is None else [x_future[i] for i in keep_indices]
    sf_bal = None if safety_future is None else [safety_future[i] for i in keep_indices]

    balanced = (
        x[keep_indices],
        x_next[keep_indices],
        safety_signal[keep_indices],
        sample_safety_value[keep_indices],
        invariant_flag[keep_indices],
        keep_indices,
        xf_bal,
        sf_bal,
    )
    if return_indices:
        return balanced

    # When indices are not requested, still return a placeholder for indices to keep tuple arity consistent.
    keep_placeholder = np.array([], dtype=int)
    return (
        balanced[0],
        balanced[1],
        balanced[2],
        balanced[3],
        balanced[4],
        keep_placeholder,
        xf_bal,
        sf_bal,
    )


def save_histogram(safety_signal_values: np.ndarray, save_path: str) -> None:
    """
    Save histogram of safety signal values.
    
    Args:
        safety_signal_values: Safety signal values
        save_path: Path to save the histogram image
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.hist(safety_signal_values, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Safety Signal Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Safety Signal Values in Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(top=2.5e5)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved histogram to {save_path}")


def print_statistics(
    n_seg_total: int,
    x_list: List[np.ndarray],
    invariant_flag_list: List[np.ndarray]
) -> None:
    """Print dataset statistics."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nSegment-level statistics:")
    print(f"  Total segments: {n_seg_total}")
    
    total_samples = sum([x.shape[0] for x in x_list])
    inv_samples = int(sum([f.sum() for f in invariant_flag_list]))
    not_inv_samples = total_samples - inv_samples
    
    print(f"\nSample-level statistics:")
    if total_samples > 0:
        pct_inv = 100 * inv_samples / total_samples
        pct_not_inv = 100 * not_inv_samples / total_samples
        print(f"  Total samples:   {total_samples:7d} | Invariant: {inv_samples:7d} ({pct_inv:5.1f}%) | Not Invariant: {not_inv_samples:7d} ({pct_not_inv:5.1f}%)")
    
    if len(x_list) > 0:
        avg_len = np.mean([x.shape[0] for x in x_list])
        print(f"  Average segment length: {avg_len:.2f}")
        
        inv_lens = [f.sum() for f in invariant_flag_list if f.sum() > 0]
        if inv_lens:
            print(f"  Average invariant segment length: {np.mean(inv_lens):.2f}")
        
        not_inv_lens = [f.shape[0] - f.sum() for f in invariant_flag_list if f.shape[0] - f.sum() > 0]
        if not_inv_lens:
            print(f"  Average not invariant segment length: {np.mean(not_inv_lens):.2f}")


def process_hj_dataset(
    safety_analysis_root: str,
    raw_dataset_name: str = "data_raw.hdf5",
    save_dataset: bool = True,
    test_proportion: float = 0.2,
    segment_by_event: bool = True,
    filter_obs_abs_threshold: float | None = None,
) -> Tuple[Tuple[List, List], str]:
    """
    Process HJ dataset for training.
    
    Args:
        safety_analysis_root: Root directory containing the raw dataset
        raw_dataset_name: Name of the raw HDF5 file
        save_dataset: Whether to save processed data to disk
        test_proportion: Proportion of data to use for test set
        filter_obs_abs_threshold: If set, drop any event segment whose policy_obs absolute value exceeds this value.
        
    Returns:
        Tuple of ((train_set, test_set), out_path)
    """
    raw_dataset_path = os.path.join(safety_analysis_root, raw_dataset_name)
    out_path = os.path.join(safety_analysis_root, "data_processed")
    
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw dataset not found: {raw_dataset_path}")
    
    # Sanity check and print data composition
    print_data_composition(raw_dataset_path)

    # Initialize accumulators (legacy outputs + per-state future trajectories)
    x_out, x_next_out, safety_signal_out, sample_safety_value_out, invariant_flag_out = [], [], [], [], []
    x_future_out: List[np.ndarray] = []
    safety_future_out: List[np.ndarray] = []

    n_seg_total = 0
    n_seg_filtered_obs = 0
    n_samples_filtered_obs = 0
    
    # Process all episodes
    with h5py.File(raw_dataset_path, "r") as f:
        raw_data = f["data"]
        
        # Get signal keys from first episode
        first_episode_key = list(raw_data.keys())[0]
        signal_keys = find_signal_keys(raw_data[first_episode_key])
        
        # Process each episode
        for episode_key in raw_data.keys():
            episode = raw_data[episode_key]
            data = load_episode_data(episode, signal_keys)
            
            # Segment by events (or keep full episode)
            safety_signal_segs, stability_segs, obs_segs = segment_by_events(
                data, use_events=segment_by_event
            )

            # Process each segment
            for safety_signal_seg, stab_seg, obs_seg in zip(
                safety_signal_segs, stability_segs, obs_segs
            ):
                if filter_obs_abs_threshold is not None and obs_seg.size > 0:
                    max_abs_obs = float(np.max(np.abs(obs_seg)))
                    if max_abs_obs > filter_obs_abs_threshold:
                        n_seg_filtered_obs += 1
                        n_samples_filtered_obs += max(0, len(obs_seg) - 1)
                        continue

                # Compute features with unbounded lookahead
                (
                    x,
                    x_next,
                    safety_signal_cur,
                    sample_safety_value,
                    invariant_flag,
                    x_future_traj,
                    l_future_traj,
                ) = process_segment(
                    safety_signal_seg, stab_seg, obs_seg
                )

                # Skip empty segments
                if x.size == 0:
                    continue
                
                # Update statistics
                n_seg_total += 1

                # Add to legacy outputs (all segments now used)
                x_out.append(x)
                x_next_out.append(x_next)
                safety_signal_out.append(safety_signal_cur)
                sample_safety_value_out.append(sample_safety_value)
                invariant_flag_out.append(invariant_flag)
                x_future_out.extend(x_future_traj)
                safety_future_out.extend(l_future_traj)
    
    # Print statistics for the base (pointwise) dataset
    print("\n====== BASE DATASET (POINTWISE) ======")
    print_statistics(
        n_seg_total,
        x_out,
        invariant_flag_out,
    )
    if filter_obs_abs_threshold is not None:
        print("\nPolicy observation outlier filter:")
        print(f"  Threshold: abs(policy_obs) <= {filter_obs_abs_threshold:g}")
        print(f"  Filtered segments: {n_seg_filtered_obs}")
        print(f"  Filtered samples:  {n_samples_filtered_obs}")

    # Concatenate all segments
    print("\nPreparing data for training...")
    x_out = np.concatenate(x_out, axis=0)
    x_next_out = np.concatenate(x_next_out, axis=0)
    safety_signal_out = np.concatenate(safety_signal_out, axis=0)
    sample_safety_value_out = np.concatenate(sample_safety_value_out, axis=0)
    invariant_flag_out = np.concatenate(invariant_flag_out, axis=0)
    
    # Save histogram before balancing
    print("\nSaving histograms...")
    hist_path = out_path + "_safety_signal_hist.png"
    save_histogram(safety_signal_out, hist_path)
    
    # Balance dataset
    print("\nBalancing base dataset...")
    (
        x_out,
        x_next_out,
        safety_signal_out,
        sample_safety_value_out,
        invariant_flag_out,
        keep_indices,
        x_future_out,
        safety_future_out,
    ) = balance_invariant_samples(
        x_out,
        x_next_out,
        safety_signal_out,
        sample_safety_value_out,
        invariant_flag_out,
        x_future=x_future_out,
        safety_future=safety_future_out,
        return_indices=True,
    )

    # Save histogram after balancing
    hist_path_balanced = out_path + "_safety_signal_hist_balanced.png"
    save_histogram(safety_signal_out, hist_path_balanced)
    
    # Print balanced statistics
    total_samples_bal = x_out.shape[0]
    inv_samples_bal = int(np.sum(invariant_flag_out))
    not_inv_samples_bal = total_samples_bal - inv_samples_bal
    pct_inv = 100 * inv_samples_bal / total_samples_bal if total_samples_bal > 0 else 0
    pct_not_inv = 100 * not_inv_samples_bal / total_samples_bal if total_samples_bal > 0 else 0
    print(f"\nBalanced base dataset:")
    print(f"  Total samples: {total_samples_bal:7d} | Invariant: {inv_samples_bal:7d} ({pct_inv:5.1f}%) | Not Invariant: {not_inv_samples_bal:7d} ({pct_not_inv:5.1f}%)")

    # Create base dataset
    print("\nCreating base dataset (pointwise)...")
    dataset = list(
        zip(
            x_out,
            x_next_out,
            safety_signal_out,
            sample_safety_value_out,
            invariant_flag_out,
            x_future_out,
            safety_future_out,
        )
    )
    
    # RNG for shuffling/balancing
    rng = np.random.default_rng()

    # ======================
    # Base train/test split
    # ======================
    n_total = len(dataset)
    n_test = int(n_total * test_proportion)
    n_train = n_total - n_test

    indices = rng.permutation(n_total)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    train_set = [dataset[i] for i in train_indices]
    test_set = [dataset[i] for i in test_indices]

    print("\n====== BASE TRAIN/TEST SPLIT ======")
    print(f"  Train set: {len(train_set)} samples ({100*len(train_set)/n_total:.1f}%)")
    print(f"  Test set:  {len(test_set)} samples ({100*len(test_set)/n_total:.1f}%)")

    # Save datasets
    if save_dataset:
        print("\nSaving base datasets...")
        train_path = out_path + "_train.pkl"
        test_path = out_path + "_test.pkl"
        
        with open(train_path, "wb") as f:
            pickle.dump(train_set, f)
        print(f"  Saved train set to {train_path}")
        
        with open(test_path, "wb") as f:
            pickle.dump(test_set, f)
        print(f"  Saved test set to {test_path}")
    
    print("\n" + "="*80)
    print("DATA PROCESSING COMPLETE")
    print("="*80 + "\n")
    
    return (train_set, test_set), out_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process safety analysis dataset for HJ value training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python safety_value/scripts/data_prep/data_processing.py \\
      --safety_analysis_root logs/safety_analysis/g1_flat_ppo_1000
        """
    )
    parser.add_argument(
        '--safety_analysis_root',
        type=str,
        required=True,
        help='Root directory containing the raw dataset (e.g., logs/safety_analysis/g1_flat_ppo_1000)'
    )
    parser.add_argument(
        '--raw_dataset_name',
        type=str,
        default="data_raw.hdf5",
        help='Raw HDF5 dataset filename (default: data_raw.hdf5)'
    )
    parser.add_argument(
        '--no_save_data',
        action='store_true',
        help='If set, do not save the processed dataset to disk'
    )
    parser.add_argument(
        '--test_proportion',
        type=float,
        default=0.2,
        help='Proportion of test set (default: 0.2)'
    )
    parser.add_argument(
        '--no_segment_by_event',
        action='store_false',
        dest='segment_by_event',
        help='If set, do not segment episodes by event; keep full episodes as single segments'
    )
    parser.add_argument(
        '--filter_obs_abs_threshold',
        type=float,
        default=None,
        help='If set, drop any event segment whose policy_obs absolute value exceeds this threshold.'
    )
    parser.add_argument(
        '--verify_saved_files',
        action='store_true',
        help='If set, reload saved pickle files after writing. Disabled by default because large datasets can exceed RAM.'
    )
    
    args = parser.parse_args()
    
    try:
        (train_set, test_set), out_path = process_hj_dataset(
            safety_analysis_root=args.safety_analysis_root,
            raw_dataset_name=args.raw_dataset_name,
            save_dataset=not args.no_save_data,
            test_proportion=args.test_proportion,
            segment_by_event=args.segment_by_event,
            filter_obs_abs_threshold=args.filter_obs_abs_threshold,
        )
        
        # Verify saved files without reloading large pickle payloads by default.
        if not args.no_save_data:
            print("Checking saved files...")
            train_pkl_path = out_path + "_train.pkl"
            test_pkl_path = out_path + "_test.pkl"

            for path in (train_pkl_path, test_pkl_path):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Saved dataset not found: {path}")
                size_gib = os.path.getsize(path) / (1024 ** 3)
                print(f"✓ Wrote {path} ({size_gib:.2f} GiB)")

            if args.verify_saved_files:
                with open(train_pkl_path, "rb") as f:
                    train_loaded = pickle.load(f)
                with open(test_pkl_path, "rb") as f:
                    test_loaded = pickle.load(f)

                print(f"✓ Successfully loaded {len(train_loaded)} train samples from {train_pkl_path}")
                print(f"✓ Successfully loaded {len(test_loaded)} test samples from {test_pkl_path}")
            print("✓ Dataset file check complete")
        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
