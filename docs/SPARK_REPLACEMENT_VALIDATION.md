# SPARK versus the original MuJoCo deployment — 2026-09-23

**The current SPARK path passes basic execution checks but does not yet qualify
as a behaviorally equivalent replacement for `unitree_mujoco-lambda`.**
The original repositories should remain as reference implementations. This
verification changed no production controller, simulator settings or learned
weights; its code and this report are staged for review.

Subsequent scope decision: the user accepts trajectory differences and keeps
the built-in 29-DoF G1. Exact native equivalence is no longer the acceptance
criterion. See [SPARK value accuracy](SPARK_VALUE_ACCURACY.md) for the new
training and held-out evaluation criterion. The measurements below remain
the historical compatibility audit.

## What was executed

The test compiled the original `unitree_rl_lab-lambda` C++ observation manager,
action manager, observation functions, ONNX actor and safety monitor. It used
the original `unitree_mujoco-lambda` robot XML and extracted the motor torque
expression verbatim from `simulate/src/unitree_sdk2_bridge.h`.

The deterministic driver supplies keyboard commands and ideal MoCap samples
directly. It steps MuJoCo synchronously and uses simulation time for the native
safety latch. Disabled LED and joystick SDK interfaces are fixture substitutes;
no DDS channels, hardware clients or network messages are opened. The build
records these substitutions and original source hashes.

This is a comparison against the original C++ computation and plant, **not a
run of the original interactive executable or a DDS latency test**. Both paths
use installed MuJoCo 3.8.0 to isolate the implementation differences. The native
controller uses its bundled ONNX Runtime 1.22.0; SPARK uses 1.29.0, and the value
comparison also exercises the current PyTorch checkpoint loader.

Four configurations ran each of ten push and ten avoidance scenarios:

1. Original C++ reference, original model, 2 ms physics.
2. SPARK, original model, 2 ms physics.
3. SPARK, original model, 5 ms physics.
4. Current SPARK default, built-in G1 model, 5 ms physics.

All use the same released actor for the selected family, a 20 ms controller
period, matched initial root poses/joint positions and deterministic command
and disturbance schedules. Episodes run for up to 1,000 control steps (20 s).
A moderate lateral velocity impulse occurs at 5 s. Seeds 0 and 5 also receive
a deliberately large 4 m/s impulse at 10 s. Avoidance includes identical
scheduled physical ball launches every 3.5 s. The original scene has no ball;
the test adds the same sphere independently to the native reference scene and
supplies its state as ideal MoCap. The native start-up suspension band and
get-up state machine are excluded.

In total: 80 cohort runs, 512 fixed-state controller samples, two repeated native
runs, and four 200-step target-replay tests. The original source repositories
were read-only. The existing 200-update value checkpoint was not retrained.

## Results

| Check | Measured result | Interpretation |
| --- | --- | --- |
| Native reference repeatability | Identical saved states, observations, actions, labels and values on repeated push/avoid seed 1 | The reference is deterministic |
| Actor inference on identical observations | Maximum difference 0 for both actors | Actor/runtime compatibility passes |
| Value inference on identical observations | Maximum difference `2.384e-7` | The learned value executes consistently in both runtimes |
| Push observation history | Maximum difference `8.941e-8` | Passes the `1e-5` observation tolerance |
| Avoidance base 480 observation entries | Maximum difference `8.941e-8` | Base-state/history layout agrees |
| Avoidance ball observations below 5 m | Maximum difference `3.338e-6` | Near-ball transformation/history agrees |
| Avoidance across near/far ball inputs | Observation difference up to `9.999`; target difference up to `0.4803` rad; value difference up to `0.9753` | The native and SPARK ball observation contracts differ |
| Identical target replay, original model, 2 ms | Generalized-position maximum difference `0.01968` after up to 200 steps | Native bridge and SPARK control timing differ |
| Same test with native sensor refresh synchronized to SPARK | Position and velocity differences exactly 0, at both 2 ms and 5 ms | Timing explains this target-replay discrepancy |
| Fall versus no-fall across the cohort | All three SPARK variants agree with the reference in all 20 scenarios | Basic outcome agreement passes for these cases |

All four configurations had two falls in ten push runs and two falls in ten
avoidance runs, following the deliberately large disturbances. Matching those
outcomes does not imply matching trajectories or monitor predictions.

The next table compares each SPARK avoidance run against the original C++
reference at matching control-step times, up to the shorter episode length.
RMSE pools all paired frames and Cartesian coordinates. Sign disagreement
counts `V(x) > 0` differently between the two runs; it is not an accuracy score.

| SPARK avoidance configuration | Root coordinate RMSE | Value RMSE | Value-sign disagreement |
| --- | --- | --- | --- |
| Original model, 2 ms | 0.5572 m | 0.2260 | 6.05% |
| Original model, 5 ms | 0.7487 m | 0.2418 | 6.30% |
| Built-in G1 model, 5 ms | 0.7311 m | 0.2433 | 5.98% |

The built-in model's push root-coordinate RMSE was 0.1644 m. These observations
establish that rollouts differ; ten deliberately scheduled cases per family do
not establish population-level reliability or statistical equivalence.

## Causes and remaining requirements

**Ball observations:** the original C++ MoCap deployment limits ball-position
norm to 5 m and zeros ball velocity outside that range. SPARK currently uses
unbounded relative position and velocity, following the original Isaac
simulation observation functions. This is a deployment-versus-training
contract difference, not an ONNX conversion problem. A native-compatible
observation profile must be explicit; silently changing the existing profile
would also change the inputs used to train the current value checkpoint.

**Physics and feedback timing:** the original XML uses MuJoCo's 2 ms default;
SPARK's migrated launcher chooses 5 ms. More fundamentally, the original bridge
computes PD torques from sensor buffers, while SPARK uses current joint state
and refreshes forward quantities after each controller step. Synchronizing the
reference sensor refresh with SPARK removes the target-replay error completely.
Exact native replay therefore needs an explicit timing profile. This test does
not characterize the additional timing variability of the real DDS loop.

**Built-in robot model:** this changes physical parameters as well as visuals.
Compiled robot mass is 35.112142 kg for the original model and 33.345331 kg for
the built-in model. Joint damping, friction, armature, geometry and actuator
configuration also differ. The built-in model should be treated as a distinct
plant and qualified with fresh rollouts and value training/evaluation.

**Safety definitions and alignment:** comparisons above use one common
post-action physical signal and fall rule on both trajectories. The native
deployment's own pre-action tilt/distance-proxy signal is saved separately.
It is not the same training label as SPARK's tilt/contact/height signal. Neither
matching runtime predictions nor matching fall counts validates the value's
safety accuracy. On these scheduled built-in-model runs, the existing value
detected only 117 of 675 frames labelled unsafe by the observed future maximum
within each launch segment (17.3% recall). That finite-horizon diagnostic is
not a claim about infinite-horizon reachability; the small checkpoint remains
unsuitable as evidence of reliable monitoring.

Before retiring the original simulator path:

1. Define and implement an original-deployment compatibility profile covering
   observation clipping, physics/control/feedback timing and signal alignment.
   Keep the built-in-model profile explicit and separate in dataset metadata.
2. Make the native observation and target-replay gates pass for that profile.
3. For the built-in model, collect representative safe and failure trajectories,
   train sufficiently, and evaluate held-out event detection and false-safe
   errors across multiple training and rollout seeds.
4. Test the real transport/start-up/interactive path if DDS, keyboard/FSM or
   MoCap IO are part of the replacement requirement. These tests do not cover
   hardware, the paper's 37-action suite or older MuJoCo versions.

## Reproduce and inspect

From `lambda_reach`, using a new output directory:

```bash
.venv/bin/python scripts/validation/verify_spark_replacement.py \
  --episodes 10 --steps 1000 \
  --output logs/safety_analysis/spark_29dof/replacement_validation/recheck
```

The command builds the native reference with `g++`, system Eigen/yaml-cpp/
spdlog/fmt, the original repo's bundled ONNX Runtime and the active MuJoCo
library. It requires both original repositories for validation, not deployment.
It writes JSON metrics, compressed traces, a plot, build logs/source hashes and
a Markdown summary. **Exit status 1 means a replacement-equivalence gate
failed**; the report still completes. Exceptions also return nonzero and must
not be mistaken for a completed audit (`results.json` must have `complete: true`).

Local results:

- [Machine-readable results](../logs/safety_analysis/spark_29dof/replacement_validation/cohort/results.json)
- [Trajectory/value plot](../logs/safety_analysis/spark_29dof/replacement_validation/cohort/comparison.png)
- [Generated report](../logs/safety_analysis/spark_29dof/replacement_validation/cohort/REPORT.md)

Results remain in the repository's ignored `logs/` directory. All maintained
verification source and documentation are staged; no commits were made.
