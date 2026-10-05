"""Run SPARK execution with a λ observer, or collect research-labelled rollouts."""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['push', 'avoid'], default='avoid')
    parser.add_argument('--backend', choices=['mujoco', 'isaac'], default='mujoco')
    parser.add_argument('--mujoco-scene', choices=['spark', 'released'], default='spark',
                        help='SPARK built-in G1 (default) or original Unitree deployment scene')
    parser.add_argument('--value', type=Path, help='Value result directory or ONNX; omit to collect training labels')
    parser.add_argument('--bundle', type=Path, help='Hash-checked SPARK policy/value bundle.json')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-output', type=Path, help='Also export the original evaluation format')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--velocity-command', type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument('--viewer', action='store_true', help='Show live values and pace execution in real time')
    parser.add_argument('--software-rendering', action='store_true',
                        help='Use Mesa software OpenGL, without shadows/reflections and with a 20 FPS display')
    parser.add_argument('--viewer-fps', type=float,
                        help='Display updates per simulated second (default: 20 with software rendering, otherwise 50); control and monitor remain 50 Hz')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    bundle_files = None
    if args.bundle:
        if args.policy or args.config or args.value:
            parser.error('--bundle supplies --policy, --config and --value together')
        if args.family != 'avoid' or args.mujoco_scene != 'spark':
            parser.error('The SPARK avoidance bundle requires --family avoid --mujoco-scene spark')
        from safety_value.monitoring.spark_bundle import load_bundle
        bundle_files = load_bundle(args.bundle)
        args.policy, args.config = bundle_files['policy'], bundle_files['config']
        args.value = bundle_files['value_config'].parent
    viewer_fps = args.viewer_fps if args.viewer_fps is not None else (20. if args.software_rendering else 50.)
    if not math.isfinite(viewer_fps) or viewer_fps <= 0:
        parser.error('--viewer-fps must be finite and positive')
    if args.software_rendering:
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'mesa'
    if args.backend != 'mujoco' and args.mujoco_scene != 'spark':
        parser.error('--mujoco-scene released requires --backend mujoco')
    app, environment = None, None
    try:
        if args.backend == 'isaac':
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=not args.viewer, device='cuda:0').app
        from spark_pipeline.locomotion import (
            make_locomotion_environment,
            runtime_versions,
        )
        from spark_pipeline.trajectory import EpisodeWriter, collect_episodes
        from spark_policy.control.whole_body.unitree_g1.locomotion import (
            RESOURCE_DIR,
            UnitreeLocomotionPolicy,
        )
        from spark_utils.assets import file_sha256, resolve_assets

        from safety_value.monitoring.spark_monitor import (
            ReachabilityObserver,
            ReachabilityViewerObserver,
            SignalConfig,
            ValuePredictor,
        )
        from safety_value.safety_analysis_algos.spark_dataset import export_legacy
        manifest = RESOURCE_DIR / 'assets.json'
        released_scene = args.backend == 'mujoco' and args.mujoco_scene == 'released'
        cache = resolve_assets(manifest, cache_dir=args.cache_dir,
                               groups=(['robot'] if released_scene else []) + ([] if args.policy else [args.family]),
                               download=not args.no_download)
        config = args.config or RESOURCE_DIR / 'configs' / f'{args.family}.yaml'
        policy_path = args.policy or cache / args.family / 'policy.onnx'
        environment, robot = make_locomotion_environment(
            config_path=config, model_path=cache / 'g1/scene_29dof.xml' if released_scene else None,
            backend=args.backend,
            seed=args.seed, max_steps=args.steps, enable_ball=args.family == 'avoid', enable_viewer=args.viewer,
            velocity_command=args.velocity_command,
            agent_options={'viewer_show_simulation_info': args.viewer,
                           'viewer_fast_render': args.software_rendering})
        policy = UnitreeLocomotionPolicy(robot, model_path=policy_path, config_path=config)
        value = ValuePredictor(args.value, checkpoint=bundle_files['value_checkpoint'] if bundle_files else None) if args.value else None
        signal_config = SignalConfig(profile=f'simulation_{args.family}')
        observer = (ReachabilityViewerObserver(environment.agent, value, config=signal_config)
                    if args.viewer else ReachabilityObserver(value, config=signal_config))
        metadata = {
            'backend': args.backend, 'task': f'g1-{args.family}', 'seed': args.seed,
            'robot_config': type(robot).__name__,
            'policy_sha256': policy.model_sha256, 'config_sha256': policy.config.source_sha256,
            'asset_manifest_sha256': file_sha256(manifest), 'signal': asdict(observer.config),
            'value_sha256': None if value is None else value.sha256,
            'physics_dt': .005, 'control_dt': .02, 'versions': runtime_versions(),
            'velocity_command': args.velocity_command,
            'termination_profile': 'unitree29-height-tilt-v1',
            'contact_semantics': 'peak substep net force; ball includes ground contacts',
            'viewer_fps': viewer_fps if args.viewer else None,
            'software_rendering': args.software_rendering,
        }
        if args.backend == 'mujoco':
            metadata.update(mujoco_scene=args.mujoco_scene,
                            mujoco_model_path=environment.agent.mujoco_xml_path,
                            mujoco_model_sha256=file_sha256(environment.agent.mujoco_xml_path))
            print(f'MuJoCo scene ({args.mujoco_scene}): {environment.agent.mujoco_xml_path}', flush=True)
        else:
            metadata['ball_max_angular_velocity_deg_s'] = environment.agent.ball_max_angular_velocity_deg_s
            metadata['allow_self_collision'] = environment.agent.allow_self_collision
        if args.viewer:
            print(f'Control and monitor: 50 Hz simulation clock; display: {viewer_fps:g} FPS. '
                  'Episode summaries report measured wall-clock frequency.', flush=True)
        with EpisodeWriter(args.output, metadata) as writer:
            summaries = collect_episodes(environment, policy, writer, episodes=args.episodes,
                                         max_steps=args.steps, observer=observer,
                                         render=args.viewer, real_time=args.viewer,
                                         render_fps=viewer_fps)
        if args.legacy_output:
            export_legacy(args.output, args.legacy_output)
        args.output.with_suffix('.summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps({'output': str(args.output), 'episodes': summaries}))
    except Exception:
        import traceback
        error = traceback.format_exc()
        print(error, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.failure.txt').write_text(error)
        raise
    finally:
        if environment is not None:
            environment.agent.close_viewer()
        if app is not None:
            app.close(wait_for_replicator=False, exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
