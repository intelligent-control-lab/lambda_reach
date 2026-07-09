#!/usr/bin/env python3
"""
Evaluate safety analysis inference results across multiple methods and experiments.

This script computes temporal recall metrics by analyzing how early the safety value
predictor can detect upcoming unsafe states compared to when they actually occur.

Temporal Recall Metrics:
    - t_unsafe: Duration from segment start to first unsafe state (safety_signal > 0)
    - t_lead: Duration from first safety_value > 0 to first safety_signal > 0
    - temporal_recall: t_lead / t_unsafe (how early the predictor warns)

Sample Value Error:
    - true_safety_value[t]: max(safety_signal[t:segment_end])
    - sample_value_error: mean squared error between predicted safety_value and true_safety_value

Usage:
    python evaluate_inference.py \
        --sa_roots g1_flat_ppo_1000 g1_flat_ppo_2000 \
        --methods dpe_lam0.9 supervised \
        --seed 42 \
        --plot_worst 5
"""

import argparse
import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


@dataclass
class SegmentMetrics:
    """Metrics for a single trajectory segment.

    The metric clock starts at ``t_event`` (the disturbance onset), not at
    ``start_step``: all of ``t_unsafe`` / ``t_lead`` / ``sample_value_error`` are
    computed over the window ``[t_event, end_step]``. In the legacy/sim "split"
    segmentation each segment starts at its event so ``t_event == start_step``
    (the pre-event window is empty) and behavior is unchanged.
    """
    episode_name: str
    segment_idx: int
    start_step: int
    end_step: int
    t_event: int             # Absolute index where the metric clock starts
    t_unsafe: Optional[int]  # Steps from t_event until safety_signal > 0
    t_lead: Optional[int]    # Steps from safety_value > 0 to safety_signal > 0
    temporal_recall: Optional[float]  # t_lead / t_unsafe
    sample_value_error: float  # MSE between predicted and true safety values

    def __repr__(self):
        recall = "None" if self.temporal_recall is None else f"{self.temporal_recall:.3f}"
        return (f"Segment(ep={self.episode_name}, seg={self.segment_idx}, "
                f"steps=[{self.start_step}:{self.end_step}], t_event={self.t_event}, "
                f"t_unsafe={self.t_unsafe}, t_lead={self.t_lead}, "
                f"recall={recall}, sve={self.sample_value_error:.4f})")


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a full episode."""
    episode_name: str
    segments: List[SegmentMetrics]
    mean_temporal_recall: Optional[float]  # Mean recall across segments with unsafe states
    
    def __repr__(self):
        return (f"Episode(name={self.episode_name}, segments={len(self.segments)}, "
                f"mean_recall={self.mean_temporal_recall:.3f if self.mean_temporal_recall else None})")


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



def find_signal_keys(episode_group: h5py.Group) -> Dict[str, str]:
    """Find signal keys in episode data using pattern matching."""
    keys = list(episode_group.keys())
    signals = {}
    
    # Find safety_signal_*
    safety_keys = [k for k in keys if k.startswith("safety_signal_")]
    if safety_keys:
        signals["safety_signal"] = safety_keys[0]
    
    # Find event_*
    event_keys = [k for k in keys if k.startswith("event_")]
    if event_keys:
        signals["event"] = event_keys[0]
    
    # Find safety_value
    if "safety_value" in keys:
        signals["safety_value"] = "safety_value"
    else:
        safety_value_keys = [k for k in keys if "safety_value" in k.lower()]
        if safety_value_keys:
            signals["safety_value"] = safety_value_keys[0]
    
    # Find stability_signal_* (optional, for segment-level invariant computation)
    stability_keys = [k for k in keys if k.startswith("stability_signal_")]
    if stability_keys:
        signals["stability_signal"] = stability_keys[0]
    
    return signals


def segment_trajectory(
    event_flag: Optional[np.ndarray],
    num_steps: int,
    mode: str = "split",
) -> List[Tuple[int, int, int]]:
    """
    Segment a trajectory into (start, end, t_event) windows (end inclusive).

    ``t_event`` is the absolute index where the metric clock starts (the
    disturbance onset); all per-segment metrics are computed over
    ``[t_event, end]``.

    mode="split" (legacy / sim): event indices are segment *boundaries*, so each
        segment starts at its event and ``t_event == start``. The pre-first-event
        block (if any) is its own segment with ``t_event == start == 0``. This
        reproduces the original sim segmentation exactly.

    mode="event_tstar" (real / annotated): the whole trajectory is ONE segment
        and ``t_event`` is the first event index (disturbance onset), or 0 if no
        event. The pre-event prefix ``[0, t_event-1]`` is excluded from metrics.
        Use this when segments were annotated as a *period containing* an event
        (longer than event->event), as in the real-hardware data.

    Args:
        event_flag: Binary event indicators (None if no events)
        num_steps: Total number of steps
        mode: "split" or "event_tstar"

    Returns:
        List of (start_step, end_step, t_event) tuples (end is inclusive)
    """
    if mode == "event_tstar":
        if event_flag is not None and np.sum(event_flag > 0.5) > 0:
            t_event = int(np.where(event_flag > 0.5)[0][0])
        else:
            t_event = 0
        return [(0, num_steps - 1, t_event)]

    # mode == "split"
    if event_flag is None or np.sum(event_flag > 0.5) == 0:
        # No events, entire trajectory is one segment
        return [(0, num_steps - 1, 0)]

    # Find event indices
    event_indices = np.where(event_flag > 0.5)[0]

    segments = []

    # First segment: from 0 to before first event (pre-roll, t_event=0)
    if event_indices[0] > 0:
        segments.append((0, event_indices[0] - 1, 0))

    # Middle segments: from each event to before the next event (t_event=event)
    for i in range(len(event_indices) - 1):
        start = event_indices[i]
        end = event_indices[i + 1] - 1
        if start < end:
            segments.append((start, end, start))

    # Last segment: from the last event to end (t_event=event)
    last_event = event_indices[-1]
    if last_event < num_steps - 1:
        segments.append((last_event, num_steps - 1, last_event))

    return segments


def compute_segment_metrics(
    episode_name: str,
    segment_idx: int,
    start: int,
    end: int,
    t_event: int,
    safety_signal: np.ndarray,
    safety_value: np.ndarray,
    value_clip: Optional[float] = None,
) -> SegmentMetrics:
    """
    Compute temporal recall metrics for a trajectory segment.

    The metric clock starts at ``t_event``: everything is computed over the
    window ``[t_event, end]`` and ``t_unsafe`` / ``t_lead`` are measured from
    ``t_event`` (the disturbance onset), not from ``start``. In legacy/sim mode
    ``t_event == start`` so the window is the whole segment (unchanged).

    Args:
        episode_name: Name of the episode
        segment_idx: Index of this segment
        start: Start step (inclusive)
        end: End step (inclusive)
        t_event: Index where the metric clock starts (start <= t_event <= end)
        safety_signal: Ground truth safety signal
        safety_value: Predicted safety value
        value_clip: If set, clip the predicted value to [-value_clip, value_clip]
            before computing the value-error MSE. The target (future-max of the
            signal) is bounded to the signal range [-1, 1] by construction, so an
            unbounded predictor (e.g. dpe) can blow up the MSE on OOD inputs;
            clipping treats overshoot as "saturated". Sign-based metrics
            (t_lead/recall) are unaffected (clip preserves sign).

    Returns:
        SegmentMetrics with computed values
    """
    # Metric window = [t_event, end]; pre-event prefix is excluded.
    segment_safety_signal = safety_signal[t_event:end+1]
    segment_safety_value = safety_value[t_event:end+1]

    # Compute true safety value: max safety signal from current step to end of segment
    segment_length = len(segment_safety_signal)
    true_safety_value = np.zeros(segment_length)
    for i in range(segment_length):
        true_safety_value[i] = np.max(segment_safety_signal[i:])

    # Compute sample value error (MSE), clipping the prediction to the bounded
    # target range if requested.
    value_for_error = segment_safety_value
    if value_clip is not None:
        value_for_error = np.clip(segment_safety_value, -value_clip, value_clip)
    sample_value_error = float(np.mean((value_for_error - true_safety_value) ** 2))

    # Find first time safety_signal > 0
    unsafe_indices = np.where(segment_safety_signal > 0)[0]

    if len(unsafe_indices) == 0:
        # Segment never becomes unsafe
        return SegmentMetrics(
            episode_name=episode_name,
            segment_idx=segment_idx,
            start_step=start,
            end_step=end,
            t_event=t_event,
            t_unsafe=None,
            t_lead=None,
            temporal_recall=None,
            sample_value_error=sample_value_error
        )

    first_unsafe_idx = unsafe_indices[0]
    t_unsafe = first_unsafe_idx  # Steps from t_event

    # Find first time safety_value > 0 at/after t_event and before first_unsafe_idx
    pred_unsafe_indices = np.where(segment_safety_value[:first_unsafe_idx+1] > 0)[0]

    if len(pred_unsafe_indices) == 0:
        # Predictor never warned before actual unsafe state
        t_lead = 0
    else:
        first_pred_unsafe_idx = pred_unsafe_indices[0]
        t_lead = first_unsafe_idx - first_pred_unsafe_idx

    # Compute temporal recall
    if t_unsafe == 0:
        # Became unsafe at t_event, no lead time possible
        temporal_recall = 0.0
    else:
        temporal_recall = t_lead / t_unsafe

    return SegmentMetrics(
        episode_name=episode_name,
        segment_idx=segment_idx,
        start_step=start,
        end_step=end,
        t_event=t_event,
        t_unsafe=t_unsafe,
        t_lead=t_lead,
        temporal_recall=temporal_recall,
        sample_value_error=sample_value_error
    )


def plot_method_comparison(
    sa_root: str,
    methods: List[str],
    episode_name: str,
    seed: int,
    base_dir: Path = Path("logs/safety_analysis"),
    subfolder: Optional[str] = None,
    segment_mode: str = "split",
    value_clip: Optional[float] = None,
):
    """
    Plot comparison of multiple methods on the same episode.
    
    Creates a figure with 2 subplots:
    - Top: Safety signal (ground truth, same for all methods)
    - Bottom: Safety values from all methods overlaid for direct comparison
    
    Args:
        sa_root: Safety analysis root directory
        methods: List of method names to compare
        episode_name: Name of the episode to plot
        seed: Random seed
        base_dir: Base directory for logs
        subfolder: Optional subfolder for organizing plots (e.g., "n_worst", "n_best")
    """
    # Load data from all methods
    method_data = {}
    
    # Colors for different methods
    method_colors = ['purple', 'orangered', 'teal', 'brown', 'darkgreen', 'navy']
    
    for method in methods:
        dataset_path = base_dir / sa_root / "results" / method / "inference" / f"seed_{seed}" / "dataset.hdf5"
        
        if not dataset_path.exists():
            continue
        
        with h5py.File(dataset_path, "r") as f:
            data_group = f["data"]
            
            if episode_name not in data_group:
                continue
            
            episode_group = data_group[episode_name]
            signal_keys = find_signal_keys(episode_group)
            
            safety_signal_key = signal_keys.get("safety_signal")
            safety_value_key = signal_keys.get("safety_value")
            event_key = signal_keys.get("event")
            
            if safety_signal_key is None or safety_value_key is None:
                continue
            
            safety_signal = np.array(episode_group[safety_signal_key])
            safety_value = np.array(episode_group[safety_value_key])
            event_flag = np.array(episode_group[event_key]) if event_key else None
            
            # Compute segments and metrics
            num_steps = len(safety_signal)
            segments = segment_trajectory(event_flag, num_steps, mode=segment_mode)

            segment_metrics = []
            for seg_idx, (start, end, t_event) in enumerate(segments):
                metrics = compute_segment_metrics(
                    episode_name, seg_idx, start, end, t_event,
                    safety_signal, safety_value, value_clip=value_clip
                )
                segment_metrics.append(metrics)
            
            method_data[method] = {
                'safety_signal': safety_signal,
                'safety_value': safety_value,
                'event_flag': event_flag,
                'segments': segments,
                'metrics': segment_metrics
            }
    
    if len(method_data) == 0:
        print(f"[WARNING] No valid data found for episode {episode_name}")
        return
    
    # Get reference data (safety_signal and events are the same for all methods)
    reference_method = next(iter(method_data.keys()))
    safety_signal = method_data[reference_method]['safety_signal']
    event_flag = method_data[reference_method]['event_flag']
    segments = method_data[reference_method]['segments']
    
    # Try to load stability signal from first method's dataset
    stability_signal = None
    dataset_path = base_dir / sa_root / "results" / reference_method / "inference" / f"seed_{seed}" / "dataset.hdf5"
    if dataset_path.exists():
        with h5py.File(dataset_path, "r") as f:
            data_group = f["data"]
            if episode_name in data_group:
                episode_group = data_group[episode_name]
                signal_keys = find_signal_keys(episode_group)
                stability_key = signal_keys.get("stability_signal")
                if stability_key:
                    stability_signal = np.array(episode_group[stability_key])
    
    steps = np.arange(len(safety_signal))
    
    # Create 2-subplot figure (smaller size)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # =============================================================================
    # Top subplot: Safety Signal (Ground Truth) with Stability Signal
    # =============================================================================
    ax = axes[0]
    ax.plot(steps, safety_signal, '-', color='blue', linewidth=3, label='Safety Signal', zorder=10)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Unsafe Threshold')
    ax.set_ylabel('Safety Signal', fontsize=16, fontweight='bold', color='blue')
    ax.set_title(f'Episode: {episode_name}', fontsize=17, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 3.0)
    ax.tick_params(axis='y', labelcolor='blue', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    
    # Add stability signal on secondary y-axis if available
    if stability_signal is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, stability_signal, '-', color='darkviolet', linewidth=2, label='Stability Signal', alpha=0.7, zorder=5)
        ax2.set_ylabel('Stability Signal', fontsize=16, fontweight='bold', color='darkviolet')
        ax2.set_ylim(-0.2, 1.2)
        ax2.tick_params(axis='y', labelcolor='darkviolet', labelsize=13)
    
    # Mark segment boundaries
    for start, end, t_event in segments:
        ax.axvline(x=start, color='gray', alpha=0.4, linestyle=':', linewidth=1.5)
        if t_event != start:
            ax.axvline(x=t_event, color='orange', alpha=0.6, linestyle='-', linewidth=2)
    
    # Mark events
    if event_flag is not None:
        event_indices = np.where(event_flag > 0.5)[0]
        for i, idx in enumerate(event_indices):
            ax.axvline(x=idx, color='orange', alpha=0.6, linestyle='-', linewidth=2.5,
                      label='Event' if i == 0 else '', zorder=5)
    
    # Shade safe/unsafe regions
    ax.fill_between(steps, safety_signal, 0, where=(safety_signal <= 0),
                    color='green', alpha=0.15, label='Safe Region')
    ax.fill_between(steps, safety_signal, 0, where=(safety_signal > 0),
                    color='red', alpha=0.15, label='Unsafe Region')
    
    # Two-column legend: data elements | regions
    signal_handles = [
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Safety Signal'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Unsafe Threshold'),
        plt.Line2D([0], [0], color='orange', linewidth=2.5, label='Event')
    ]
    if stability_signal is not None:
        signal_handles.append(plt.Line2D([0], [0], color='darkviolet', linewidth=2, label='Stability Signal'))
    
    region_handles = [
        plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.15, label='Safe Region'),
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.15, label='Unsafe Region')
    ]
    ax.legend(handles=signal_handles + region_handles, loc='upper left', 
             fontsize=12, ncol=2, framealpha=0.95, columnspacing=1.5)
    
    # =============================================================================
    # Bottom subplot: Safety Values (All Methods Overlaid)
    # =============================================================================
    ax = axes[1]
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Safety Value', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 3.0)
    ax.tick_params(labelsize=13)
    
    # Plot each method's safety value
    method_handles = []
    annotations_data = []  # Store annotation positions to check for overlaps
    
    for idx, (method, data) in enumerate(sorted(method_data.items())):
        color = method_colors[idx % len(method_colors)]
        safety_value = data['safety_value']
        metrics = data['metrics']
        
        # Plot the safety value line
        line = ax.plot(steps, safety_value, '-', color=color, linewidth=2.5, 
               label=f'{method}', alpha=0.85, zorder=10)[0]
        method_handles.append(line)
        
        # Mark first predictions and collect annotation data
        for seg_idx, ((start, end, t_event), metric) in enumerate(zip(data['segments'], metrics)):
            if metric.t_unsafe is not None and metric.t_lead is not None and metric.t_lead > 0:
                unsafe_step = t_event + metric.t_unsafe
                pred_step = unsafe_step - metric.t_lead
                
                # Mark prediction time with a marker
                ax.plot(pred_step, safety_value[pred_step], 'o', color=color, 
                       markersize=9, markeredgecolor='white', markeredgewidth=2, zorder=15)
                
                # Store data for annotation (will be placed after all data is collected)
                annotations_data.append({
                    'pred_step': pred_step,
                    'value': safety_value[pred_step],
                    'recall': metric.temporal_recall,
                    'color': color,
                    'method_idx': idx,
                    'seg_idx': seg_idx
                })
    
    # Now place all annotations with overlap avoidance
    # Start with upper left, then adjust locally if overlaps detected
    placed_boxes = []  # List of (x_min, x_max, y_min, y_max) for placed annotations
    
    for ann_data in annotations_data:
        pred_step = ann_data['pred_step']
        value = ann_data['value']
        recall = ann_data['recall']
        color = ann_data['color']
        
        # Initial position: upper left, closer to the marker
        x_offset = -60
        y_offset = 35
        
        # Check for overlaps and adjust position
        attempts = 0
        max_attempts = 20
        while attempts < max_attempts:
            # Estimate bounding box in data coordinates (approximate)
            # Typical annotation box is about 80 steps wide and 0.4 units tall
            ann_x = pred_step + x_offset * (steps[-1] - steps[0]) / 1000  # Convert offset to data coords
            ann_y = value + y_offset * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 400
            box_width = 80
            box_height = 0.35
            
            # Check if this box overlaps with any existing box
            overlap = False
            for existing_box in placed_boxes:
                ex_min, ex_max, ey_min, ey_max = existing_box
                # Check for overlap
                if not (ann_x + box_width < ex_min or ann_x > ex_max or 
                       ann_y + box_height < ey_min or ann_y > ey_max):
                    overlap = True
                    break
            
            if not overlap:
                # No overlap, place the annotation here
                placed_boxes.append((ann_x, ann_x + box_width, ann_y, ann_y + box_height))
                break
            
            # Overlap detected, try a different position
            attempts += 1
            if attempts < 5:
                # Try moving more to the left
                x_offset -= 25
            elif attempts < 10:
                # Try moving down
                y_offset -= 15
            elif attempts < 15:
                # Try moving up more
                y_offset += 20
            else:
                # Try moving right
                x_offset += 30
        
        # Place the annotation
        annotation_text = f"$R_{{\\mathrm{{temp}}}}$ = {recall:.2f}"
        ax.annotate(annotation_text, 
                   xy=(pred_step, value),
                   xytext=(x_offset, y_offset), textcoords='offset points',
                   fontsize=11, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.9, edgecolor=color, linewidth=1.5),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Mark segment boundaries
    for start, end, t_event in segments:
        ax.axvline(x=start, color='gray', alpha=0.4, linestyle=':', linewidth=1.5)
        if t_event != start:
            ax.axvline(x=t_event, color='orange', alpha=0.6, linestyle='-', linewidth=2)
    
    # Mark events
    if event_flag is not None:
        event_indices = np.where(event_flag > 0.5)[0]
        for i, idx in enumerate(event_indices):
            ax.axvline(x=idx, color='orange', alpha=0.6, linestyle='-', linewidth=2.5, zorder=5)
    
    # Two-column legend: methods in first column | markers in second column
    marker_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=9, markeredgecolor='white', markeredgewidth=2,
                  label='First Prediction', linestyle='None')
    ]
    
    # Arrange legend: methods | markers
    all_handles = method_handles + marker_handles
    ax.legend(handles=all_handles, loc='upper left', 
             fontsize=12, ncol=2, framealpha=0.95, columnspacing=1.5)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    # Save plot under results/_inference_comparison/seed_{seed}/plots/
    # Filename includes seed info: seed_{seed}_{episode}_inference_comparison.png
    if subfolder:
        output_dir = base_dir / sa_root / "results" / "_inference_comparison" / f"seed_{seed}" / "plots" / subfolder
    else:
        output_dir = base_dir / sa_root / "results" / "_inference_comparison" / f"seed_{seed}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{seed}_{episode_name}_inference_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_segment_analysis(
    episode_name: str,
    steps: np.ndarray,
    safety_signal: np.ndarray,
    safety_value: np.ndarray,
    event_flag: Optional[np.ndarray],
    segments: List[Tuple[int, int]],
    metrics: List[SegmentMetrics],
    output_path: str,
    stability_signal: Optional[np.ndarray] = None
):
    """
    Plot safety signals with segment boundaries and temporal metrics.
    
    Args:
        episode_name: Name of the episode
        steps: Step indices
        safety_signal: Ground truth safety signal
        safety_value: Predicted safety value
        event_flag: Event indicators (or None)
        segments: List of (start, end) segment boundaries
        metrics: List of computed metrics for each segment
        output_path: Path to save the plot
        stability_signal: Optional stability signal for dual y-axis plotting
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    
    # Plot 1: Safety signal with t_unsafe markers and stability signal
    ax = axes[0]
    ax.plot(steps, safety_signal, '-', color='blue', linewidth=2.5, label='Safety Signal')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Unsafe Threshold')
    ax.set_ylabel('Safety Signal', fontsize=16, fontweight='bold', color='blue')
    ax.set_title(f'Episode: {episode_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 3.0)
    ax.tick_params(axis='y', labelcolor='blue', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    
    # Add stability signal on secondary y-axis if available
    if stability_signal is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, stability_signal, '-', color='darkviolet', linewidth=2, label='Stability Signal', alpha=0.7)
        ax2.set_ylabel('Stability Signal', fontsize=16, fontweight='bold', color='darkviolet')
        ax2.set_ylim(-0.2, 1.2)
        ax2.tick_params(axis='y', labelcolor='darkviolet', labelsize=13)
    
    # Mark segment boundaries and t_unsafe
    for seg_idx, ((start, end, t_event), metric) in enumerate(zip(segments, metrics)):
        # Segment boundary
        ax.axvline(x=start, color='gray', alpha=0.5, linestyle=':', linewidth=1.5)
        # Event onset (metric clock start); only visible when t_event != start
        if t_event != start:
            ax.axvline(x=t_event, color='orange', alpha=0.7, linestyle='-', linewidth=2)

        if metric.t_unsafe is not None:
            unsafe_step = t_event + metric.t_unsafe
            ax.axvline(x=unsafe_step, color='darkred', alpha=0.7, linestyle='--', linewidth=2)
            ax.plot(unsafe_step, safety_signal[unsafe_step], 'ro', markersize=10)
            
            # Annotate t_unsafe
            ax.annotate(f't_unsafe={metric.t_unsafe:.1f}', 
                       xy=(unsafe_step, safety_signal[unsafe_step]),
                       xytext=(8, 15), textcoords='offset points',
                       fontsize=11, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    # Mark events if available
    if event_flag is not None:
        event_indices = np.where(event_flag > 0.5)[0]
        for idx in event_indices:
            ax.axvline(x=idx, color='orange', alpha=0.6, linestyle='-', linewidth=2.5)
    
    # Create legend
    signal_handles = [
        plt.Line2D([0], [0], color='blue', linewidth=2.5, label='Safety Signal'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Unsafe Threshold'),
        plt.Line2D([0], [0], color='orange', linewidth=2.5, label='Events')
    ]
    if stability_signal is not None:
        signal_handles.append(plt.Line2D([0], [0], color='darkviolet', linewidth=2, label='Stability Signal'))
    ax.legend(handles=signal_handles, loc='upper left', fontsize=12, ncol=1, framealpha=0.95)
    
    # Plot 2: Safety value with t_lead markers
    ax = axes[1]
    ax.plot(steps, safety_value, '-', color='green', linewidth=2.5, label='Safety Value')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Safety Value', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 3.0)
    ax.tick_params(labelsize=13)
    
    # Mark segment boundaries and t_lead
    for seg_idx, ((start, end, t_event), metric) in enumerate(zip(segments, metrics)):
        # Segment boundary
        ax.axvline(x=start, color='gray', alpha=0.5, linestyle=':', linewidth=1.5)
        if t_event != start:
            ax.axvline(x=t_event, color='orange', alpha=0.7, linestyle='-', linewidth=2)

        if metric.t_unsafe is not None and metric.t_lead is not None and metric.t_lead > 0:
            unsafe_step = t_event + metric.t_unsafe
            pred_step = unsafe_step - metric.t_lead
            
            # Mark prediction time (first warning)
            ax.axvline(x=pred_step, color='darkgreen', alpha=0.7, linestyle='--', linewidth=2)
            ax.plot(pred_step, safety_value[pred_step], 'o', color='darkgreen', markersize=10)
            
            # Mark actual unsafe time
            ax.axvline(x=unsafe_step, color='darkred', alpha=0.7, linestyle='--', linewidth=2)
            
            # Annotate t_lead with arrow
            mid_step = (pred_step + unsafe_step) / 2
            ax.annotate('', xy=(unsafe_step, safety_value[pred_step]), 
                       xytext=(pred_step, safety_value[pred_step]),
                       arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2.5))
            ax.annotate(f't_lead={metric.t_lead:.1f}', 
                       xy=(mid_step, safety_value[pred_step]),
                       xytext=(0, 12), textcoords='offset points',
                       ha='center', fontsize=12, color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))
            
            # Annotate recall
            ax.annotate(f'recall={metric.temporal_recall:.2f}', 
                       xy=(pred_step, safety_value[pred_step]),
                       xytext=(0, -25), textcoords='offset points',
                       ha='center', fontsize=12, color='blue', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.85))
    
    # Mark events if available
    if event_flag is not None:
        event_indices = np.where(event_flag > 0.5)[0]
        for idx in event_indices:
            ax.axvline(x=idx, color='orange', alpha=0.6, linestyle='-', linewidth=2.5)
    
    # Create two-column legend
    value_handles = [
        plt.Line2D([0], [0], color='green', linewidth=2.5, label='Safety Value'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                   markersize=10, label='First Prediction', linestyle='None')
    ]
    ax.legend(handles=value_handles, loc='upper left', fontsize=12, ncol=1, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_episode(
    episode_group: h5py.Group,
    episode_name: str,
    output_dir: Optional[Path] = None,
    segment_mode: str = "split",
    value_clip: Optional[float] = None,
) -> EpisodeMetrics:
    """
    Evaluate a single episode.

    Args:
        episode_group: HDF5 group containing episode data
        episode_name: Name of the episode
        output_dir: Directory to save plots (None to skip plotting)
        segment_mode: "split" (sim) or "event_tstar" (real); see segment_trajectory

    Returns:
        EpisodeMetrics with all segments and mean recall
    """
    # Find signal keys
    signal_keys = find_signal_keys(episode_group)
    
    # Load data
    safety_signal_key = signal_keys.get("safety_signal")
    safety_value_key = signal_keys.get("safety_value")
    event_key = signal_keys.get("event")
    stability_key = signal_keys.get("stability_signal")
    
    if safety_signal_key is None or safety_value_key is None:
        return EpisodeMetrics(episode_name=episode_name, segments=[], mean_temporal_recall=None)
    
    safety_signal = np.array(episode_group[safety_signal_key])
    safety_value = np.array(episode_group[safety_value_key])
    event_flag = np.array(episode_group[event_key]) if event_key else None
    stability_signal = np.array(episode_group[stability_key]) if stability_key else None
    
    num_steps = len(safety_signal)

    # Segment trajectory
    segments = segment_trajectory(event_flag, num_steps, mode=segment_mode)

    # Compute metrics for each segment
    all_metrics = []
    for seg_idx, (start, end, t_event) in enumerate(segments):
        metrics = compute_segment_metrics(
            episode_name, seg_idx, start, end, t_event,
            safety_signal, safety_value, value_clip=value_clip
        )
        all_metrics.append(metrics)
    
    # Compute mean temporal recall for this episode
    valid_recalls = [m.temporal_recall for m in all_metrics if m.temporal_recall is not None]
    mean_recall = np.mean(valid_recalls) if len(valid_recalls) > 0 else None
    
    # Plot if output directory provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        steps = np.arange(num_steps)
        output_path = output_dir / f"{episode_name}_segment_analysis.png"
        plot_segment_analysis(
            episode_name, steps, safety_signal, safety_value,
            event_flag, segments, all_metrics, str(output_path),
            stability_signal=stability_signal
        )
    
    return EpisodeMetrics(
        episode_name=episode_name,
        segments=all_metrics,
        mean_temporal_recall=mean_recall
    )


def evaluate_method(
    sa_root: str,
    method: str,
    seed: int,
    base_dir: Path = Path("logs/safety_analysis"),
    episodes_to_plot: Optional[set] = None,
    skip_evaluation: bool = False,
    plot_subfolder: Optional[str] = None,
    segment_mode: str = "split",
    value_clip: Optional[float] = None,
) -> Tuple[List[EpisodeMetrics], Path]:
    """
    Evaluate all episodes for a given method and seed.
    
    Args:
        sa_root: Safety analysis root directory name
        method: Method subdirectory name
        seed: Random seed
        base_dir: Base directory for safety analysis logs
        episodes_to_plot: If provided, plot only these episode names
        skip_evaluation: If True, skip evaluation phase and only do plotting
        plot_subfolder: Optional subfolder name for plots (e.g., "n_worst", "n_best")
        
    Returns:
        (List of episode metrics, path to dataset directory)
    """
    # Construct path to dataset
    dataset_dir = base_dir / sa_root / "results" / method / "inference" / f"seed_{seed}"
    dataset_path = dataset_dir / "dataset.hdf5"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    all_episode_metrics = []
    
    # Phase 1: Evaluate all episodes (no plotting) - only if not skipped
    if not skip_evaluation:
        print(f"\n{'='*80}")
        print(f"Evaluating: {sa_root} / {method} / seed_{seed}")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_path}")
        print("Phase 1: Evaluating all episodes...")
        
        with h5py.File(dataset_path, "r") as f:
            data_group = f["data"]
            episode_names = list(data_group.keys())
            print(f"Found {len(episode_names)} episodes")
            
            for ep_name in tqdm(episode_names, desc="Evaluating episodes"):
                episode_group = data_group[ep_name]
                ep_metrics = evaluate_episode(episode_group, ep_name, output_dir=None,
                                              segment_mode=segment_mode, value_clip=value_clip)
                all_episode_metrics.append(ep_metrics)
        
        # Count total segments
        total_segments = sum(len(ep.segments) for ep in all_episode_metrics)
        print(f"Evaluated {total_segments} segments across {len(episode_names)} episodes")
    
    # Phase 2: Plot specified episodes if requested
    if episodes_to_plot is not None and len(episodes_to_plot) > 0:
        if skip_evaluation:
            # When skipping evaluation, we still need to load episode metrics for the ones to plot
            available_episodes = episodes_to_plot
        else:
            # Filter to episodes that exist in our data
            available_episodes = {ep.episode_name for ep in all_episode_metrics}
            episodes_to_plot_filtered = episodes_to_plot & available_episodes
        
        episodes_to_plot_filtered = episodes_to_plot if skip_evaluation else episodes_to_plot & available_episodes
        
        if len(episodes_to_plot_filtered) > 0:
            if not skip_evaluation:
                print(f"\nPhase 2: Plotting {len(episodes_to_plot_filtered)} specified episodes...")
            
            # Determine plot directory based on subfolder
            if plot_subfolder:
                plot_dir = dataset_dir / "plots" / plot_subfolder
            else:
                plot_dir = dataset_dir / "plots"
            
            with h5py.File(dataset_path, "r") as f:
                data_group = f["data"]
                
                for ep_name in tqdm(sorted(episodes_to_plot_filtered), desc="Plotting episodes"):
                    if ep_name in data_group:
                        episode_group = data_group[ep_name]
                        # Re-evaluate with plotting enabled
                        evaluate_episode(episode_group, ep_name, output_dir=plot_dir,
                                         segment_mode=segment_mode, value_clip=value_clip)
            
            print(f"Plots saved to: {plot_dir}")
    
    return all_episode_metrics, dataset_dir


def compute_confusion_matrix_from_dataset(
    dataset_path: Path,
    segment_mode: str = "split",
) -> ConfusionMatrix:
    """
    Compute confusion matrix for invariant prediction across all episodes.

    Segment-level invariant label computation (following data_processing.py logic):
    - A state is invariant if the maximum future safety signal (inclusive) stays non-positive
    - Each timestep receives its own invariant label based on this criterion
    - Only the per-segment metric window [t_event, end] is scored; the pre-event
      prefix is excluded (in "split" mode the windows tile the whole episode, so
      coverage is unchanged).

    Prediction is based on safety_value sign:
    - prediction = 1 if safety_value <= 0 (predicts safe/invariant)
    - prediction = 0 if safety_value > 0 (predicts unsafe)

    Args:
        dataset_path: Path to HDF5 dataset file
        segment_mode: "split" (sim) or "event_tstar" (real); see segment_trajectory

    Returns:
        ConfusionMatrix with statistics
    """
    tp = fp = tn = fn = 0

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        episode_names = list(data_group.keys())

        for ep_name in tqdm(episode_names, desc="Computing confusion matrix", leave=False):
            episode_group = data_group[ep_name]

            # Find signal keys
            signal_keys = find_signal_keys(episode_group)
            safety_signal_key = signal_keys.get("safety_signal")
            safety_value_key = signal_keys.get("safety_value")
            event_key = signal_keys.get("event")

            if safety_signal_key is None or safety_value_key is None:
                continue

            safety_signal = np.array(episode_group[safety_signal_key])
            safety_value = np.array(episode_group[safety_value_key])
            event_flag = np.array(episode_group[event_key]) if event_key else None

            num_steps = len(safety_signal)

            # Segment trajectory by events
            segments = segment_trajectory(event_flag, num_steps, mode=segment_mode)

            # Score each segment over its metric window [t_event, end].
            for start, end, t_event in segments:
                w_sig = safety_signal[t_event:end+1]
                w_val = safety_value[t_event:end+1]
                if len(w_sig) == 0:
                    continue
                future_max = np.maximum.accumulate(w_sig[::-1])[::-1]
                inv = (future_max <= 0).astype(int)        # 1=invariant(safe), 0=will-be-unsafe
                pred = (w_val <= 0).astype(int)            # 1=predicts safe
                tp += int(np.sum((pred == 1) & (inv == 1)))
                fp += int(np.sum((pred == 1) & (inv == 0)))
                tn += int(np.sum((pred == 0) & (inv == 0)))
                fn += int(np.sum((pred == 0) & (inv == 1)))

    return ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn
    )


def compute_statistics(episode_metrics_list: List[EpisodeMetrics]) -> Dict:
    """
    Compute summary statistics from episode metrics.
    
    Args:
        episode_metrics_list: List of all episode metrics
        
    Returns:
        Dictionary of statistics
    """
    # Flatten all segments
    all_segments = []
    for ep in episode_metrics_list:
        all_segments.extend(ep.segments)
    
    # Filter segments that have valid metrics (became unsafe)
    valid_metrics = [m for m in all_segments if m.temporal_recall is not None]
    
    # Compute sample value error statistics (for all segments)
    sample_value_errors = np.array([m.sample_value_error for m in all_segments])
    
    if len(valid_metrics) == 0:
        return {
            "total_segments": len(all_segments),
            "segments_with_unsafe": 0,
            "segments_safe": len(all_segments),
            "temporal_recall_mean": None,
            "temporal_recall_std": None,
            "temporal_recall_min": None,
            "temporal_recall_max": None,
            "temporal_recall_median": None,
            "t_unsafe_mean": None,
            "t_lead_mean": None,
            "detection_rate": 0.0,
            "sample_value_error_mean": float(np.mean(sample_value_errors)),
            "sample_value_error_std": float(np.std(sample_value_errors)),
            "sample_value_error_min": float(np.min(sample_value_errors)),
            "sample_value_error_max": float(np.max(sample_value_errors)),
            "sample_value_error_median": float(np.median(sample_value_errors)),
        }
    
    recalls = np.array([m.temporal_recall for m in valid_metrics])
    t_unsafes = np.array([m.t_unsafe for m in valid_metrics])
    t_leads = np.array([m.t_lead for m in valid_metrics])
    
    # Detection rate: fraction of segments where predictor warned (t_lead > 0)
    detection_rate = np.sum(t_leads > 0) / len(t_leads)
    
    return {
        "total_segments": len(all_segments),
        "segments_with_unsafe": len(valid_metrics),
        "segments_safe": len(all_segments) - len(valid_metrics),
        "temporal_recall_mean": float(np.mean(recalls)),
        "temporal_recall_std": float(np.std(recalls)),
        "temporal_recall_min": float(np.min(recalls)),
        "temporal_recall_max": float(np.max(recalls)),
        "temporal_recall_median": float(np.median(recalls)),
        "t_unsafe_mean": float(np.mean(t_unsafes)),
        "t_lead_mean": float(np.mean(t_leads)),
        "detection_rate": float(detection_rate),
        "sample_value_error_mean": float(np.mean(sample_value_errors)),
        "sample_value_error_std": float(np.std(sample_value_errors)),
        "sample_value_error_min": float(np.min(sample_value_errors)),
        "sample_value_error_max": float(np.max(sample_value_errors)),
        "sample_value_error_median": float(np.median(sample_value_errors)),
    }


def print_results_table(results: Dict[Tuple[str, str], Dict]):
    """
    Print results in a formatted table.
    
    Args:
        results: Dictionary mapping (sa_root, method) to statistics
    """
    print("\n" + "="*130)
    print("EVALUATION RESULTS SUMMARY")
    print("="*130)
    
    # Header - show: Recall, DetRate, t_lead, t_unsafe, SVE, Inv.Acc
    print(f"{'SA Root':<25} {'Method':<20} "
          f"{'Recall':<10} {'DetRate':<10} {'t_lead':<9} {'t_unsafe':<9} {'SVE':<10} {'Inv.Acc':<10}")
    print("-"*130)
    
    # Sort results by sa_root, then method
    sorted_keys = sorted(results.keys())
    
    for (sa_root, method) in sorted_keys:
        stats = results[(sa_root, method)]
        
        inv_acc = stats.get("invariant_accuracy", float('nan'))
        
        if stats["temporal_recall_mean"] is not None:
            print(f"{sa_root:<25} {method:<20} "
                  f"{stats['temporal_recall_mean']:<10.4f} "
                  f"{stats['detection_rate']:<10.2%} "
                  f"{stats['t_lead_mean']:<9.2f} "
                  f"{stats['t_unsafe_mean']:<9.2f} "
                  f"{stats['sample_value_error_mean']:<10.4f} "
                  f"{inv_acc:<10.4f}")
        else:
            print(f"{sa_root:<25} {method:<20} "
                  f"{'N/A':<10} {'N/A':<10} {'N/A':<9} {'N/A':<9} "
                  f"{stats['sample_value_error_mean']:<10.4f} "
                  f"{inv_acc:<10.4f}")
    
    print("="*130)
    
    # Print detailed statistics
    print("\nDETAILED STATISTICS:")
    print("="*140)
    
    for (sa_root, method) in sorted_keys:
        stats = results[(sa_root, method)]
        print(f"\n{sa_root} / {method}:")
        print(f"  Total segments: {stats['total_segments']}")
        print(f"  Segments with unsafe states: {stats['segments_with_unsafe']}")
        print(f"  Segments always safe: {stats['segments_safe']}")
        
        if stats['temporal_recall_mean'] is not None:
            print(f"  Temporal Recall:")
            print(f"    Mean:   {stats['temporal_recall_mean']:.4f}")
            print(f"    Std:    {stats['temporal_recall_std']:.4f}")
            print(f"    Min:    {stats['temporal_recall_min']:.4f}")
            print(f"    Max:    {stats['temporal_recall_max']:.4f}")
            print(f"    Median: {stats['temporal_recall_median']:.4f}")
            print(f"  Detection Rate: {stats['detection_rate']:.2%}")
            print(f"  Average t_lead: {stats['t_lead_mean']:.2f} steps")
            print(f"  Average t_unsafe: {stats['t_unsafe_mean']:.2f} steps")
        else:
            print(f"  No unsafe segments to evaluate")
        
        print(f"  Sample Value Error (MSE):")
        print(f"    Mean:   {stats['sample_value_error_mean']:.4f}")
        print(f"    Std:    {stats['sample_value_error_std']:.4f}")
        print(f"    Min:    {stats['sample_value_error_min']:.4f}")
        print(f"    Max:    {stats['sample_value_error_max']:.4f}")
        print(f"    Median: {stats['sample_value_error_median']:.4f}")
        
        # Print confusion matrix statistics
        if "confusion_matrix" in stats:
            cm = stats["confusion_matrix"]
            print(f"  Invariant Prediction (Confusion Matrix):")
            print(f"    Total samples:   {cm['total']:,}")
            print(f"    True Positives:  {cm['true_positives']:,}")
            print(f"    False Positives: {cm['false_positives']:,}")
            print(f"    True Negatives:  {cm['true_negatives']:,}")
            print(f"    False Negatives: {cm['false_negatives']:,}")
            print(f"    Accuracy:  {stats['invariant_accuracy']:.4f}")
            print(f"    Precision: {stats['invariant_precision']:.4f}")
            print(f"    Recall:    {stats['invariant_recall']:.4f}")
            print(f"    F1 Score:  {stats['invariant_f1_score']:.4f}")
            print(f"    False Positive Rate: {stats['invariant_fpr']:.4f}")
            print(f"    False Negative Rate: {stats['invariant_fnr']:.4f}")


def save_results_to_csv(results: Dict[Tuple[str, str], Dict], output_path: Path):
    """
    Save all evaluation results to a CSV file.
    
    Args:
        results: Dictionary mapping (sa_root, method) to statistics
        output_path: Path to save the CSV file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort results by sa_root, then method
    sorted_keys = sorted(results.keys())
    
    # Define CSV headers - all available metrics
    headers = [
        'sa_root',
        'method',
        'total_segments',
        'segments_with_unsafe',
        'segments_safe',
        'temporal_recall_mean',
        'temporal_recall_std',
        'temporal_recall_min',
        'temporal_recall_max',
        'temporal_recall_median',
        'detection_rate',
        't_lead_mean',
        't_unsafe_mean',
        'sample_value_error_mean',
        'sample_value_error_std',
        'sample_value_error_min',
        'sample_value_error_max',
        'sample_value_error_median',
        'invariant_accuracy',
        'invariant_precision',
        'invariant_recall',
        'invariant_f1_score',
        'invariant_fpr',
        'invariant_fnr',
        'confusion_matrix_tp',
        'confusion_matrix_fp',
        'confusion_matrix_tn',
        'confusion_matrix_fn',
        'confusion_matrix_total',
    ]
    
    # Write CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for (sa_root, method) in sorted_keys:
            stats = results[(sa_root, method)]
            
            # Create row with all data
            row = {
                'sa_root': sa_root,
                'method': method,
                'total_segments': stats['total_segments'],
                'segments_with_unsafe': stats['segments_with_unsafe'],
                'segments_safe': stats['segments_safe'],
                'temporal_recall_mean': stats['temporal_recall_mean'] if stats['temporal_recall_mean'] is not None else '',
                'temporal_recall_std': stats['temporal_recall_std'] if stats['temporal_recall_std'] is not None else '',
                'temporal_recall_min': stats['temporal_recall_min'] if stats['temporal_recall_min'] is not None else '',
                'temporal_recall_max': stats['temporal_recall_max'] if stats['temporal_recall_max'] is not None else '',
                'temporal_recall_median': stats['temporal_recall_median'] if stats['temporal_recall_median'] is not None else '',
                'detection_rate': stats['detection_rate'],
                't_lead_mean': stats['t_lead_mean'] if stats['t_lead_mean'] is not None else '',
                't_unsafe_mean': stats['t_unsafe_mean'] if stats['t_unsafe_mean'] is not None else '',
                'sample_value_error_mean': stats['sample_value_error_mean'],
                'sample_value_error_std': stats['sample_value_error_std'],
                'sample_value_error_min': stats['sample_value_error_min'],
                'sample_value_error_max': stats['sample_value_error_max'],
                'sample_value_error_median': stats['sample_value_error_median'],
                'invariant_accuracy': stats.get('invariant_accuracy', ''),
                'invariant_precision': stats.get('invariant_precision', ''),
                'invariant_recall': stats.get('invariant_recall', ''),
                'invariant_f1_score': stats.get('invariant_f1_score', ''),
                'invariant_fpr': stats.get('invariant_fpr', ''),
                'invariant_fnr': stats.get('invariant_fnr', ''),
            }
            
            # Add confusion matrix data if available
            if 'confusion_matrix' in stats:
                cm = stats['confusion_matrix']
                row['confusion_matrix_tp'] = cm['true_positives']
                row['confusion_matrix_fp'] = cm['false_positives']
                row['confusion_matrix_tn'] = cm['true_negatives']
                row['confusion_matrix_fn'] = cm['false_negatives']
                row['confusion_matrix_total'] = cm['total']
            else:
                row['confusion_matrix_tp'] = ''
                row['confusion_matrix_fp'] = ''
                row['confusion_matrix_tn'] = ''
                row['confusion_matrix_fn'] = ''
                row['confusion_matrix_total'] = ''
            
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate safety analysis inference results with temporal recall metrics."
    )
    parser.add_argument(
        "--sa_roots",
        type=str,
        nargs="+",
        required=True,
        help="List of safety analysis root directories (e.g., g1_flat_ppo_1000)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        help="List of methods to evaluate (e.g., dpe_lam0.9 supervised)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference data (default: 42)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs/safety_analysis",
        help="Base directory for safety analysis logs (default: logs/safety_analysis)",
    )
    parser.add_argument(
        "--segment_mode",
        type=str,
        default="split",
        choices=["split", "event_tstar"],
        help=("How to segment each episode and where the metric clock starts. "
              "'split' (default, sim): events are segment boundaries, t_event=start. "
              "'event_tstar' (real/annotated): one segment per episode, t_event=event "
              "onset; metrics computed over [t_event, end] (pre-event prep excluded)."),
    )
    parser.add_argument(
        "--value_clip",
        type=float,
        default=None,
        help=("If set, clip the predicted safety value to [-V, V] before the value-error "
              "(SVE) MSE. The future-max target is bounded to the signal range [-1,1], so an "
              "unbounded predictor (dpe) can blow up SVE on OOD real obs; use --value_clip 1.0. "
              "Sign-based metrics (recall/FPR) are unaffected."),
    )
    parser.add_argument(
        "--plot_worst",
        type=int,
        default=None,
        help="Number of worst episodes (lowest temporal recall) to plot (default: None, no plotting)",
    )
    parser.add_argument(
        "--plot_best",
        type=int,
        default=None,
        help="Number of best episodes (highest temporal recall) to plot (default: None, uses same as --plot_worst if specified)",
    )
    
    args = parser.parse_args()
    
    # If plot_best not specified but plot_worst is, use same value
    if args.plot_best is None and args.plot_worst is not None:
        args.plot_best = args.plot_worst
    
    base_dir = Path(args.base_dir)
    
    print("="*120)
    print("SAFETY ANALYSIS INFERENCE EVALUATION")
    print("="*120)
    print(f"SA Roots: {args.sa_roots}")
    print(f"Methods: {args.methods}")
    print(f"Seed: {args.seed}")
    print(f"Base directory: {base_dir}")
    if args.plot_worst:
        print(f"Plot worst N episodes: {args.plot_worst}")
    if args.plot_best:
        print(f"Plot best N episodes: {args.plot_best}")
    if not args.plot_worst and not args.plot_best:
        print(f"Plotting: Disabled (use --plot_worst N and/or --plot_best N to enable)")
    print("="*120)
    
    # Evaluate all combinations
    results = {}
    confusion_matrices = {}  # (sa_root, method) -> ConfusionMatrix
    
    # Group methods by sa_root for union computation
    sa_root_methods = {}
    for sa_root in args.sa_roots:
        sa_root_methods[sa_root] = args.methods
    
    # Phase 1: Evaluate all methods without plotting
    all_evaluations = {}  # (sa_root, method) -> episode_metrics
    
    for sa_root in args.sa_roots:
        for method in args.methods:
            print(f"\n[Phase 1] Evaluating {sa_root} / {method}...")
            episode_metrics, dataset_dir = evaluate_method(
                sa_root, method, args.seed, base_dir, episodes_to_plot=None,
                segment_mode=args.segment_mode, value_clip=args.value_clip
            )
            all_evaluations[(sa_root, method)] = episode_metrics
            stats = compute_statistics(episode_metrics)
            results[(sa_root, method)] = stats

            # Compute confusion matrix for invariant prediction
            print(f"  Computing confusion matrix for invariant prediction...")
            dataset_path = dataset_dir / "dataset.hdf5"
            confusion_matrix = compute_confusion_matrix_from_dataset(dataset_path,
                                                                     segment_mode=args.segment_mode)
            confusion_matrices[(sa_root, method)] = confusion_matrix
            
            # Add confusion matrix metrics to results
            results[(sa_root, method)].update({
                "confusion_matrix": {
                    "true_positives": confusion_matrix.true_positives,
                    "false_positives": confusion_matrix.false_positives,
                    "true_negatives": confusion_matrix.true_negatives,
                    "false_negatives": confusion_matrix.false_negatives,
                    "total": confusion_matrix.total,
                },
                "invariant_accuracy": confusion_matrix.accuracy,
                "invariant_precision": confusion_matrix.precision,
                "invariant_recall": confusion_matrix.recall,
                "invariant_f1_score": confusion_matrix.f1_score,
                "invariant_fpr": confusion_matrix.false_positive_rate,
                "invariant_fnr": confusion_matrix.false_negative_rate,
            })
    
    # Phase 2: For each sa_root, find union of worst/best N episodes across all methods
    if args.plot_worst or args.plot_best:
        print(f"\n{'='*120}")
        print("IDENTIFYING EPISODES FOR PLOTTING")
        print(f"{'='*120}")
        
        for sa_root in args.sa_roots:
            # Track worst and best episodes separately
            union_worst_episodes = set()
            union_best_episodes = set()
            
            # Collect worst N episodes from each method
            if args.plot_worst:
                print(f"\n[{sa_root}] Finding union of worst {args.plot_worst} episodes across methods...")
                
                for method in args.methods:
                    episode_metrics = all_evaluations[(sa_root, method)]
                    
                    # Filter episodes with valid recall
                    episodes_with_recall = [ep for ep in episode_metrics if ep.mean_temporal_recall is not None]
                    
                    if len(episodes_with_recall) > 0:
                        # Sort by mean temporal recall (ascending - worst first)
                        episodes_with_recall.sort(key=lambda ep: ep.mean_temporal_recall)
                        
                        # Select worst N
                        worst_n = min(args.plot_worst, len(episodes_with_recall))
                        worst_episodes = episodes_with_recall[:worst_n]
                        worst_episode_names = {ep.episode_name for ep in worst_episodes}
                        
                        print(f"  [{method}] Worst {worst_n} episodes: {sorted(worst_episode_names)}")
                        union_worst_episodes.update(worst_episode_names)
                
                print(f"  [UNION] Total unique worst episodes to plot: {len(union_worst_episodes)}")
                print(f"          Episode names: {sorted(union_worst_episodes)}")
            
            # Collect best N episodes from each method
            if args.plot_best:
                print(f"\n[{sa_root}] Finding union of best {args.plot_best} episodes across methods...")
                
                for method in args.methods:
                    episode_metrics = all_evaluations[(sa_root, method)]
                    
                    # Filter episodes with valid recall
                    episodes_with_recall = [ep for ep in episode_metrics if ep.mean_temporal_recall is not None]
                    
                    if len(episodes_with_recall) > 0:
                        # Sort by mean temporal recall (descending - best first)
                        episodes_with_recall.sort(key=lambda ep: ep.mean_temporal_recall, reverse=True)
                        
                        # Select best N
                        best_n = min(args.plot_best, len(episodes_with_recall))
                        best_episodes = episodes_with_recall[:best_n]
                        best_episode_names = {ep.episode_name for ep in best_episodes}
                        
                        print(f"  [{method}] Best {best_n} episodes: {sorted(best_episode_names)}")
                        union_best_episodes.update(best_episode_names)
                
                print(f"  [UNION] Total unique best episodes to plot: {len(union_best_episodes)}")
                print(f"          Episode names: {sorted(union_best_episodes)}")
            
            # Phase 3: Plot worst episodes for all methods under this sa_root
            if union_worst_episodes:
                print(f"\n[Phase 2a] Plotting {len(union_worst_episodes)} worst episodes for all methods under {sa_root}...")
                for method in args.methods:
                    print(f"  Plotting worst for {sa_root} / {method}...")
                    evaluate_method(
                        sa_root, method, args.seed, base_dir,
                        episodes_to_plot=union_worst_episodes,
                        skip_evaluation=True,  # Skip re-evaluation, only plot
                        plot_subfolder="n_worst",
                        segment_mode=args.segment_mode, value_clip=args.value_clip
                    )

                # Create comparison plots for worst episodes
                print(f"\n[Phase 2a-comp] Creating method comparison plots for worst episodes in {sa_root}...")
                for ep_name in tqdm(sorted(union_worst_episodes), desc="Creating worst comparison plots"):
                    plot_method_comparison(sa_root, args.methods, ep_name, args.seed, base_dir,
                                           subfolder="n_worst", segment_mode=args.segment_mode,
                                           value_clip=args.value_clip)
                
                comparison_dir = base_dir / sa_root / "results" / "_inference_comparison" / f"seed_{args.seed}" / "plots" / "n_worst"
                print(f"  Worst comparison plots saved to: {comparison_dir}")
            
            # Phase 4: Plot best episodes for all methods under this sa_root
            if union_best_episodes:
                print(f"\n[Phase 2b] Plotting {len(union_best_episodes)} best episodes for all methods under {sa_root}...")
                for method in args.methods:
                    print(f"  Plotting best for {sa_root} / {method}...")
                    evaluate_method(
                        sa_root, method, args.seed, base_dir,
                        episodes_to_plot=union_best_episodes,
                        skip_evaluation=True,  # Skip re-evaluation, only plot
                        plot_subfolder="n_best",
                        segment_mode=args.segment_mode, value_clip=args.value_clip
                    )

                # Create comparison plots for best episodes
                print(f"\n[Phase 2b-comp] Creating method comparison plots for best episodes in {sa_root}...")
                for ep_name in tqdm(sorted(union_best_episodes), desc="Creating best comparison plots"):
                    plot_method_comparison(sa_root, args.methods, ep_name, args.seed, base_dir,
                                           subfolder="n_best", segment_mode=args.segment_mode,
                                           value_clip=args.value_clip)
                
                comparison_dir = base_dir / sa_root / "results" / "_inference_comparison" / f"seed_{args.seed}" / "plots" / "n_best"
                print(f"  Best comparison plots saved to: {comparison_dir}")
    
    # Print results table
    print_results_table(results)
    
    # Save results to CSV for each sa_root
    for sa_root in args.sa_roots:
        # Filter results for this sa_root
        sa_root_results = {k: v for k, v in results.items() if k[0] == sa_root}
        
        # Save to CSV under results/_inference_comparison/
        csv_path = base_dir / sa_root / "results" / "_inference_comparison" / f"seed_{args.seed}" / f"inference_evaluation_seed_{args.seed}.csv"
        save_results_to_csv(sa_root_results, csv_path)
    
    print("\n" + "="*120)
    print("EVALUATION COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()
