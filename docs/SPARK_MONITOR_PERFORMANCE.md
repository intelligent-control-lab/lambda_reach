# Online monitor performance

Measured locally on 2026-09-23 using the built-in SPARK 29-DoF G1, seed 62,
0.3 m/s forward command, released actor, SPARK-trained seed-0 value and HDF5
recording. Policy/value CPU inference used one thread. The viewer used Mesa
software OpenGL because the local NVIDIA driver/library installation could not
create the native GLX viewer.

The qualified SPARK-trained pair was checked again on 2026-10-05: headless
MuJoCo monitoring averaged 223.5 control steps per wall second. A live
100-step software-rendered launcher check reached 49.9 control Hz with a
20 FPS display. Scalar Isaac monitoring averaged 32.6 wall-clock control Hz;
its simulation still uses 50 Hz control. The exact pair and traces are linked
from the [October verification report](SPARK_TRAINING_PARITY.md). The detailed
September profiling below concerns the earlier pair.

## Diagnosis

The configured physics period is 5 ms (200 Hz in simulation time), with four
substeps per policy/value update: 20 ms (50 Hz). These configured rates do not
guarantee the same throughput in wall-clock time.

| Configuration | Steps | Measured control updates/s | Simulation / wall time |
| --- | ---: | ---: | ---: |
| Original software viewer, monitor enabled | 200 | 7.24 | 0.145x |
| Original software viewer, learned value disabled | 100 | 7.24 | 0.145x |
| Headless, monitor and recording enabled | 200 | 49.97 | 0.999x |
| Shadows/reflections disabled, display every control step | 200 | 26.07 | 0.521x |
| Shadows/reflections disabled, display at 20 FPS | 500 | 49.98 | 1.000x |
| Same configuration, longer verification | 1,000 | 49.98 | 1.000x |

Original mean times per control update:

| Operation | Mean latency |
| --- | ---: |
| Policy | 0.52 ms |
| Physics and task | 0.79 ms |
| Learned value inference | 0.37 ms |
| Complete monitor, including value inference | 0.42 ms |
| HDF5 append | 5.13 ms |
| Viewer render call, including synchronization | 130.38 ms |

Within the render call, `MujocoAgent._update_simulation_info_overlay()` spent
126.67 ms in the native viewer's `set_texts` path, and `viewer.sync()` took
3.65 ms. This is a graphics synchronization bottleneck; the long overlay timing
does not mean evaluating the value network takes 127 ms. Disabling value
inference left throughput unchanged. Disabling shadows/reflections substantially
reduced the wait; spacing display updates further removed most of it.

## Runtime changes

`--software-rendering` now selects a 20 FPS presentation cadence and disables
viewer shadows/reflections. The built-in model, meshes, dynamics, policy and
value checkpoints remain the same. The collector continues every policy,
physics, monitor and recording update; it skips only intermediate display
updates. This is Python simulation-clock pacing, not a hard real-time guarantee,
and 20 FPS presentation should not be confused with a 50 FPS viewer.

The 1,000-step verification completed 20 simulated seconds in 20.010 wall
seconds. Control intervals averaged 20.000 ms (95th percentile 20.757 ms,
maximum 23.647 ms). It recorded 1,000 observations and values and 1,001 states.
All 20 recorded state/action/observation/signal/value datasets matched the
original run exactly over the first 200 transitions. The 33 targeted SPARK and
research integration tests passed, including preservation of every control and
monitor sample when limiting display updates. Tests used
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated system ROS pytest plugin
whose `lark` dependency is unavailable in this environment.
The actual launcher also completed two 250-step episodes at 49.98 Hz each and
exited cleanly; its log is `latency/launcher_verification.log` under the same
experiment root.

```bash
bash deploy/avoid/run_spark_monitor.sh --software-rendering --velocity-command 0.3 0 0
```

Override display frequency with `--viewer-fps N`. Without software rendering the
display defaults to 50 FPS. Per-episode summary JSON now reports `wall_seconds`,
`simulation_seconds`, `actual_control_hz` and `real_time_factor`. The summary
excludes environment/reset/initial-display setup and includes control-loop
pacing and end-of-episode HDF5 flushing.

## Reproduce the profiling

Run from `lambda_reach`, choosing a new output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/validation/profile_spark_monitor.py \
  --output logs/safety_analysis/spark_29dof/latency/my_check \
  --steps 1000 --fast-visuals --render-fps 20
```

The profiler always selects software OpenGL. Omit `--fast-visuals --render-fps 20`
to reproduce the original viewer behavior; `--headless`, `--no-value`, and
`--no-record` isolate the corresponding costs. Run comparisons sequentially to
avoid competing for CPU resources. It saves a JSON report, raw timings in NPZ,
and a diagnostic trajectory. Its whole-collection frequency includes initial
episode/display setup, unlike the per-episode summary.

Raw measurements are retained locally under
`logs/safety_analysis/spark_29dof/latency/{baseline,no_value,headless,fast_visuals,fast_20fps,verified_20fps}/`.
