#!/usr/bin/env python3
"""
Per-segment diagnostic plots for the real-data inference metrics.

For each (method, segment) this draws the safety signal and the predicted safety
value side by side and overlays *exactly* the quantities that the aggregate
metrics in evaluate_inference.py are built from, so you can eyeball whether the
per-segment metric is reasonable.

Metric clock = t_event (the annotated disturbance onset). All metrics are
computed over the window [t_event, end]; the pre-event prefix [0, t_event) is
prep/standing and is EXCLUDED (greyed out). In sim, t_event=0 so the window is
the whole segment. This matches evaluate_inference.py --segment_mode event_tstar.

  - Temporal recall  -> marks t_event, the t_unsafe window (t_event -> first
    safety_signal>0) and the t_lead window (first safety_value>0 after t_event,
    before the unsafe step -> the unsafe step); annotates R_temp = t_lead/t_unsafe.

  - Sample value error -> overlays the regression target true_safety_value[t] =
    max(safety_signal[t:end]) over [t_event,end] (the future-max step function)
    and annotates the per-segment MSE.

  - FPR -> the confusion matrix labels every step in [t_event,end] by future_max>0
    ("will become unsafe" = negative class) and predicts unsafe where value>0. A
    classification strip colours each step TP/FP/TN/FN; the box annotates
    per-segment FPR = FP/(FP+TN).

Convention (matches evaluate_inference.compute_confusion_matrix_from_dataset):
    positive class = "invariant/safe" (prediction=1 when value<=0). Hence
    FP = will-be-unsafe step predicted safe (a MISS); the code's "FPR" is the
    miss-rate over will-be-unsafe steps -- not a false-alarm rate.

Usage (bash, inside the IsaacLab/hj venv):
    python safety_value/scripts/3_result_analysis/plot_segment_diagnostics.py \
        --sa_roots real_push real_avoid \
        --methods lambda_reachability_lambda_reach_lambda_0_99 supervised dpe \
        --seed 0 [--unsafe_only] [--segments demo_push_1_seg03 ...]
"""
import argparse
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ALL_METHODS = [
    "lambda_reachability_lambda_reach_lambda_0_99",
    "lambda_reachability_lambda_reach_lambda_0_95",
    "lambda_reachability_lambda_reach_lambda_0_50",
    "lambda_reachability_lambda_reach_lambda_0_00",
    "dpe",
    "supervised",
]

# Classification-strip colours (safety-detection language).
C_TN = "#2ca02c"   # will-be-unsafe & value>0   -> correctly flagged ("caught")
C_FP = "#d62728"   # will-be-unsafe & value<=0  -> missed (code's False Positive)
C_TP = "#cfe8cf"   # recoverable   & value<=0   -> correct safe (code's True Positive)
C_FN = "#ff9800"   # recoverable   & value>0    -> false alarm (code's False Negative)
C_PRE = "#bdbdbd"  # pre-event prefix (excluded from metrics)


def find_signal_keys(g: h5py.Group) -> Dict[str, str]:
    keys = list(g.keys())
    out = {}
    sk = [k for k in keys if k.startswith("safety_signal_")]
    if sk:
        out["safety_signal"] = sk[0]
    ek = [k for k in keys if k.startswith("event_")]
    if ek:
        out["event"] = ek[0]
    out["safety_value"] = "safety_value" if "safety_value" in keys else None
    return out


def value_ylim(values: np.ndarray) -> tuple:
    """Robust y-limits for the value panel; caps blown-up (e.g. dpe) values."""
    lo = min(float(np.min(values)), -1.5)
    hi = max(float(np.max(values)), 1.0)
    lo = max(lo, -3.0)   # cap absurd ranges; actual range printed in the box
    hi = min(hi, 3.5)
    return lo - 0.1, hi + 0.1


def ax_sig_ylim(sig: np.ndarray) -> tuple:
    lo = min(-1.2, float(sig.min()) - 0.1)
    hi = max(1.5, float(sig.max()) + 0.3)
    return lo, hi


def _hex(h: str):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def plot_segment(seg_name: str, method: str, sa_root: str,
                 safety_signal: np.ndarray, safety_value: np.ndarray,
                 t_event: int, out_path: Path, value_clip=None):
    T = len(safety_signal)
    steps = np.arange(T)
    t_event = int(np.clip(t_event, 0, T - 1))

    # ---- metric quantities over the window [t_event, end] (mirror evaluate_inference) ----
    w = slice(t_event, T)
    sig_w = safety_signal[w]
    val_w = safety_value[w]
    # future-max target = true_safety_value[t] = max(signal[t:end]) within window
    true_w = np.maximum.accumulate(sig_w[::-1])[::-1]
    val_for_err = val_w if value_clip is None else np.clip(val_w, -value_clip, value_clip)
    sample_value_error = float(np.mean((val_for_err - true_w) ** 2))

    will_be_unsafe = true_w > 0          # negative class (label==0), indexed within window
    pred_safe = val_w <= 0               # prediction==1 (predicts safe)
    tp = int(np.sum(pred_safe & ~will_be_unsafe))
    fp = int(np.sum(pred_safe & will_be_unsafe))     # will-be-unsafe & value<=0 (miss)
    tn = int(np.sum(~pred_safe & will_be_unsafe))    # will-be-unsafe & value>0  (caught)
    fn = int(np.sum(~pred_safe & ~will_be_unsafe))   # recoverable & value>0     (false alarm)
    neg = fp + tn
    fpr = fp / neg if neg > 0 else float("nan")

    # temporal recall (indices measured from t_event)
    unsafe_idx = np.where(sig_w > 0)[0]
    if len(unsafe_idx) > 0:
        first_unsafe = int(unsafe_idx[0])              # offset from t_event
        t_unsafe = first_unsafe
        pred_pos = np.where(val_w[:first_unsafe + 1] > 0)[0]
        if len(pred_pos) == 0:
            first_pred, t_lead = None, 0
        else:
            first_pred = int(pred_pos[0])
            t_lead = first_unsafe - first_pred
        recall = 0.0 if t_unsafe == 0 else t_lead / t_unsafe
        first_unsafe_abs = t_event + first_unsafe
        first_pred_abs = (t_event + first_pred) if first_pred is not None else None
    else:
        first_unsafe = first_pred = first_unsafe_abs = first_pred_abs = None
        t_unsafe = t_lead = recall = None

    # ----------------------------- figure -----------------------------
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(3, 1, height_ratios=[3, 3, 0.45], hspace=0.12)
    ax_sig = fig.add_subplot(gs[0])
    ax_val = fig.add_subplot(gs[1], sharex=ax_sig)
    ax_strip = fig.add_subplot(gs[2], sharex=ax_sig)

    sig_lo, sig_hi = ax_sig_ylim(safety_signal)

    # ===== Panel 1: safety signal (ground truth) =====
    ax_sig.plot(steps, safety_signal, "-", color="tab:blue", lw=2.2, label="safety_signal", zorder=6)
    ax_sig.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
    # shade will-be-unsafe vs recoverable over the metric window
    xw = steps[t_event:]
    ax_sig.fill_between(xw, sig_lo, sig_hi, where=will_be_unsafe, color="red", alpha=0.07, zorder=0)
    ax_sig.fill_between(xw, sig_lo, sig_hi, where=~will_be_unsafe, color="green", alpha=0.07, zorder=0)
    # grey out the excluded pre-event prefix
    if t_event > 0:
        ax_sig.axvspan(0, t_event, color=C_PRE, alpha=0.35, zorder=1)
    ax_sig.set_ylim(sig_lo, sig_hi)
    ax_sig.set_ylabel("safety signal", fontsize=12, fontweight="bold")
    ax_sig.set_title(f"{sa_root}  /  {method}\nsegment: {seg_name}   (T={T}, t_event={t_event})",
                     fontsize=12, fontweight="bold")
    ax_sig.grid(alpha=0.3)

    # event marker (metric clock start)
    ax_sig.axvline(t_event, color="orange", ls="-", lw=2.2, zorder=7)
    ax_sig.text(t_event, sig_hi, " t_event", color="darkorange", fontsize=10,
                fontweight="bold", va="top", ha="left")

    if first_unsafe_abs is not None:
        ax_sig.axvline(first_unsafe_abs, color="darkred", ls="--", lw=2)
        ax_sig.plot(first_unsafe_abs, safety_signal[first_unsafe_abs], "o", color="darkred", ms=9, zorder=8)
        yb = sig_lo + 0.12 * (sig_hi - sig_lo)
        ax_sig.annotate("", xy=(first_unsafe_abs, yb), xytext=(t_event, yb),
                        arrowprops=dict(arrowstyle="<->", color="darkred", lw=2))
        ax_sig.text((t_event + first_unsafe_abs) / 2, yb + 0.04 * (sig_hi - sig_lo),
                    f"$t_{{unsafe}}$ = {t_unsafe}", ha="center", va="bottom",
                    fontsize=11, color="darkred", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax_sig.legend(handles=[
        plt.Line2D([0], [0], color="tab:blue", lw=2.2, label="safety_signal"),
        plt.Line2D([0], [0], color="orange", lw=2.2, label="t_event (clock start)"),
        Patch(fc="red", alpha=0.12, label="will-be-unsafe (future_max>0)"),
        Patch(fc="green", alpha=0.12, label="recoverable (future_max≤0)"),
        Patch(fc=C_PRE, alpha=0.4, label="pre-event (excluded)"),
    ], loc="upper left", fontsize=8.5, framealpha=0.95, ncol=2)

    # ===== Panel 2: safety value (prediction) + future-max target =====
    ylo, yhi = value_ylim(np.concatenate([val_w, true_w]))
    ax_val.plot(xw, true_w, "-", color="gray", lw=1.8, alpha=0.8,
                label="true value = future-max(signal)  [SVE target]", zorder=4)
    ax_val.plot(steps, np.clip(safety_value, ylo, yhi), "-", color="tab:purple",
                lw=2.2, label="safety_value (pred)", zorder=6)
    ax_val.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
    if t_event > 0:
        ax_val.axvspan(0, t_event, color=C_PRE, alpha=0.35, zorder=1)
    ax_val.axvline(t_event, color="orange", ls="-", lw=2.2, zorder=7)
    ax_val.set_ylim(ylo, yhi)
    ax_val.set_ylabel("safety value", fontsize=12, fontweight="bold")
    ax_val.grid(alpha=0.3)

    if first_unsafe_abs is not None:
        ax_val.axvline(first_unsafe_abs, color="darkred", ls="--", lw=2)
        if first_pred_abs is not None and t_lead > 0:
            ax_val.axvline(first_pred_abs, color="darkgreen", ls="--", lw=2)
            yb = ylo + 0.82 * (yhi - ylo)
            ax_val.annotate("", xy=(first_unsafe_abs, yb), xytext=(first_pred_abs, yb),
                            arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
            ax_val.text((first_pred_abs + first_unsafe_abs) / 2, yb + 0.03 * (yhi - ylo),
                        f"$t_{{lead}}$ = {t_lead}", ha="center", va="bottom",
                        fontsize=11, color="darkgreen", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    recall_str = "n/a (no unsafe step)" if recall is None else f"{t_lead}/{t_unsafe} = {recall:.2f}"
    fpr_str = "n/a (no neg steps)" if neg == 0 else f"{fp}/({fp}+{tn}) = {fpr:.2f}"
    actual_rng = f"[{safety_value.min():.2f}, {safety_value.max():.3g}]"
    box = (f"$R_{{temp}}$ = $t_{{lead}}/t_{{unsafe}}$ = {recall_str}\n"
           f"value err (MSE vs future-max) = {sample_value_error:.3f}\n"
           f"FPR = FP/(FP+TN) = {fpr_str}\n"
           f"  FP = will-be-unsafe & value≤0 (missed)\n"
           f"value range {actual_rng}   (metrics over [t_event,end])")
    ax_val.text(0.985, 0.04, box, transform=ax_val.transAxes, ha="right", va="bottom",
                fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.95))
    ax_val.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # ===== Panel 3: per-step classification strip =====
    strip = np.zeros((1, T, 3))
    for t in range(T):
        if t < t_event:
            strip[0, t] = _hex(C_PRE)
            continue
        i = t - t_event
        if will_be_unsafe[i]:
            strip[0, t] = _hex(C_FP) if pred_safe[i] else _hex(C_TN)
        else:
            strip[0, t] = _hex(C_TP) if pred_safe[i] else _hex(C_FN)
    ax_strip.imshow(strip, aspect="auto", extent=[0, T, 0, 1], interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("time step (within segment)", fontsize=12, fontweight="bold")
    ax_strip.set_ylabel("class", fontsize=9)
    ax_strip.legend(handles=[
        Patch(fc=C_FP, label=f"FP (will-be-unsafe, value≤0 → miss)  ={fp}"),
        Patch(fc=C_TN, label=f"TN (will-be-unsafe, value>0 → caught)  ={tn}"),
        Patch(fc=C_TP, label=f"TP (recoverable, value≤0)  ={tp}"),
        Patch(fc=C_FN, label=f"FN (recoverable, value>0 → false alarm)  ={fn}"),
        Patch(fc=C_PRE, label="pre-event (excluded)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=3, fontsize=8.5, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sa_roots", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_dir", default="logs/safety_analysis")
    ap.add_argument("--unsafe_only", action="store_true",
                    help="only plot segments that contain an unsafe step (safety_signal>0)")
    ap.add_argument("--segments", nargs="+", default=None,
                    help="restrict to these segment names")
    ap.add_argument("--value_clip", type=float, default=None,
                    help="clip predicted value to [-V,V] before the SVE annotation (e.g. 1.0)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    n_total = 0
    for sa_root in args.sa_roots:
        for method in args.methods:
            ds = base / sa_root / "results" / method / "inference" / f"seed_{args.seed}" / "dataset.hdf5"
            if not ds.exists():
                print(f"[skip] {ds} not found")
                continue
            out_dir = ds.parent / "segment_diagnostics"
            n = 0
            with h5py.File(ds, "r") as f:
                for ep in f["data"].keys():
                    if args.segments and ep not in args.segments:
                        continue
                    g = f["data"][ep]
                    sk = find_signal_keys(g)
                    if sk.get("safety_signal") is None or sk.get("safety_value") is None:
                        continue
                    sig = np.asarray(g[sk["safety_signal"]], np.float64)
                    sv = np.asarray(g["safety_value"], np.float64)
                    # t_event = first event index (disturbance onset), else 0
                    t_event = 0
                    if sk.get("event") is not None:
                        ev = np.where(np.asarray(g[sk["event"]]) > 0.5)[0]
                        if len(ev) > 0:
                            t_event = int(ev[0])
                    if args.unsafe_only and not (sig > 0).any():
                        continue
                    plot_segment(ep, method, sa_root, sig, sv, t_event, out_dir / f"{ep}.png",
                                 value_clip=args.value_clip)
                    n += 1
            print(f"[{sa_root}/{method}] wrote {n} plots -> {out_dir}")
            n_total += n
    print(f"\ndone: {n_total} segment diagnostic plots")


if __name__ == "__main__":
    main()
