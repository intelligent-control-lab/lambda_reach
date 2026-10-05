"""Measure online monitor latency, including the real viewer and HDF5 writer."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-value", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--fast-visuals", action="store_true")
    parser.add_argument("--render-fps", type=float)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.render_fps is not None and (
        not np.isfinite(args.render_fps) or args.render_fps <= 0
    ):
        parser.error("--render-fps must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
    import torch
    from spark_pipeline.locomotion import make_locomotion_environment
    from spark_pipeline.trajectory import EpisodeWriter, collect_episodes
    from spark_policy.control.whole_body.unitree_g1.locomotion import (
        RESOURCE_DIR,
        UnitreeLocomotionPolicy,
    )
    from spark_utils.assets import resolve_assets

    from safety_value.monitoring.spark_monitor import (
        ReachabilityObserver,
        ReachabilityViewerObserver,
        ValuePredictor,
    )

    timings = defaultdict(list)
    control_starts = []

    def timed(owner, name, label):
        original = getattr(owner, name)

        def call(*pos, **kw):
            started = time.perf_counter()
            if label == "policy":
                control_starts.append(started)
            try:
                return original(*pos, **kw)
            finally:
                timings[label].append(time.perf_counter() - started)

        setattr(owner, name, call)

    config = RESOURCE_DIR / "configs/avoid.yaml"
    cache = resolve_assets(
        RESOURCE_DIR / "assets.json", groups=["avoid"], download=False
    )
    setup_start = time.perf_counter()
    env, robot = make_locomotion_environment(
        config_path=config,
        seed=62,
        max_steps=args.steps,
        enable_ball=True,
        velocity_command=(0.3, 0.0, 0.0),
        enable_viewer=not args.headless,
        agent_options={
            "viewer_show_simulation_info": not args.headless,
            "viewer_fast_render": args.fast_visuals,
        },
    )
    policy = UnitreeLocomotionPolicy(
        robot, model_path=cache / "avoid/policy.onnx", config_path=config
    )
    value = (
        None
        if args.no_value
        else ValuePredictor(
            Path(__file__).resolve().parents[2]
            / "logs/safety_analysis/spark_29dof/value_accuracy/lambda_reachability_seed0"
        )
    )
    observer = (
        ReachabilityObserver(value)
        if args.headless
        else ReachabilityViewerObserver(env.agent, value)
    )
    timed(policy, "act", "policy")
    timed(env, "step_output", "physics_and_task")
    timed(ValuePredictor, "__call__", "value_inference")
    timed(type(observer), "__call__", "monitor_total")
    timed(env.agent, "render", "render_total")
    timed(env.agent, "_restore_or_remember_viewer_state", "viewer_state")
    timed(env.agent, "_update_simulation_info_overlay", "viewer_overlay")
    if env.agent.renderer:
        timed(env.agent.renderer, "update_scene", "offscreen_scene")
    if env.agent.viewer:
        timed(env.agent.viewer, "sync", "viewer_sync")
    timed(EpisodeWriter, "append", "recording")
    setup_seconds = time.perf_counter() - setup_start

    class NullWriter:
        def begin(self, *args):
            pass

        def append(self, *args, **kwargs):
            pass

        def end(self):
            pass

        def close(self):
            pass

    writer = (
        NullWriter()
        if args.no_record
        else EpisodeWriter(args.output / "trajectory.hdf5", {"diagnostic": True})
    )
    started = time.perf_counter()
    try:
        summaries = collect_episodes(
            env,
            policy,
            writer,
            episodes=1,
            max_steps=args.steps,
            observer=observer,
            render=not args.headless,
            real_time=True,
            render_fps=args.render_fps,
        )
        wall = time.perf_counter() - started
    finally:
        writer.close()
        env.agent.close_viewer()
    report = {
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "renderer_environment": {
            key: os.environ.get(key)
            for key in (
                "LIBGL_ALWAYS_SOFTWARE",
                "__GLX_VENDOR_LIBRARY_NAME",
                "LP_NUM_THREADS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "setup_seconds": setup_seconds,
        "loop_wall_seconds": wall,
        "steps": summaries[0]["steps"],
        "actual_control_hz": summaries[0]["steps"] / wall,
        "simulation_speed": summaries[0]["steps"] * 0.02 / wall,
        "timings": {},
    }
    timings["control_interval"] = np.diff(control_starts).tolist()
    for name, samples in timings.items():
        if not samples:
            continue
        ms = np.array(samples) * 1000
        report["timings"][name] = {
            "calls": len(samples),
            "mean_ms": float(ms.mean()),
            "p50_ms": float(np.median(ms)),
            "p95_ms": float(np.quantile(ms, 0.95)),
            "max_ms": float(ms.max()),
        }
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.output / "timings.npz", **{k: np.array(v) for k, v in timings.items()}
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
