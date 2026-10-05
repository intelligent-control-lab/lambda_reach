# Retrain the SPARK policy and its reachability monitor

The maintained 29-DoF workflow uses only `spark_dev` and `lambda_reach`.
SPARK owns Isaac PPO, the built-in robot, disturbances, inference and MuJoCo
execution. This repository trains and evaluates the λ value using SPARK
rollouts. The old Unitree repositories and `source/hj_humanoid` are optional
historical references. No released policy is needed for these commands.

Within SPARK, `lab/spark_lab` owns reusable training machinery. G1 task/PPO
configuration and workflow assembly live under
`example/unitree_g1/locomotion/training/` and `recipes/`. The command paths and
checkpoint format below are unchanged. Runtime monitoring does not require
the optional training library.

Use this repository's `spark` branch with
[SPARK master at `fd8e4df` or later](https://github.com/YIFANSUN98/spark_dev/commit/fd8e4df8272956a2091fd5a4f602576f86edbf8b).
The qualified walking and avoidance actors are included in SPARK under
`example/unitree_g1/locomotion/pretrained/`; training a monitor for the bundled
avoidance actor does not require retraining its policy. Monitor checkpoints and
datasets remain local run artifacts, outside Git.

The [2026-09-24 execution report](SPARK_RETRAINING_RESULTS.md) records the actual
SPARK-only retraining and held-out monitor results, including that earlier
actor's quality limitation. The [October training audit](SPARK_TRAINING_PARITY.md)
records the qualified nominal and avoidance actors, their training lineage,
the new monitor's held-out accuracy and existing-simulator regression checks.
The current default viewer uses
`logs/safety_analysis/spark_29dof/spark_replacement_20261005_v2/monitor/bundle.json`.
Both legacy Unitree checkouts are archived outside the active project folder.

## Train a policy in SPARK

Create the training overlay as described in SPARK's
[training guide](https://github.com/YIFANSUN98/spark_dev/blob/master/example/unitree_g1/locomotion/TRAINING.md).
From `spark_dev`, with a working Isaac CUDA environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-locomotion/bin/python example/unitree_g1/locomotion/train_walking.py \
  --output outputs/locomotion/my_nominal_policy
```

This trains fresh standing and walking PPO teachers, then distills both into
one actor under `my_nominal_policy/actor`. The walking teacher uses an observable
gait clock. Teacher checks and a held-out standing/forward/backward gate in
both simulators must pass; failures stop the workflow. No released weights
are used. Checkpoints, ONNX/config exports, recipes, curves, source snapshots
and qualification reports stay in the output directory. The continuation
below can initialize avoidance from this actor, then train the matching monitor.

## Train a monitor from that exact policy

From `lambda_reach`, after running `scripts/setup_spark_env.sh`:

```bash
SPARK_ROOT=/path/to/spark_dev
POLICY_RUN="$SPARK_ROOT/example/unitree_g1/locomotion/pretrained/avoidance"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/train_spark_monitor.py \
  --policy "$POLICY_RUN/policy.onnx" --config "$POLICY_RUN/deploy.yaml" \
  --output logs/safety_analysis/spark_29dof/my_spark_monitor
```

For your own newly trained avoidance actor, set `POLICY_RUN` to its exported
run directory instead. Collection always uses the exact supplied policy/config.

The default is 1,638 training, 410 validation and 512 test episodes, each up to
20 seconds. Seeds are separated by whole episode before labels are balanced;
only the training anchors are balanced. The script trains λ and supervised
models from scratch at seeds 0, 1 and 2, each for 2,000 updates. λ uses two
256/256 critics with λ=δ=0.99. `--seed N` selects one training seed;
`--seeds 0 1 2` and `--no-baselines` explicitly control the experiment.

Validation must pass temporal recall ≥95%, false-safe rate ≤10%, false-alarm
rate ≤15%, and sample-value error ≤0.20. Among passing λ models, the script
chooses the lowest validation sample-value error and freezes its identity in
`selection.json` before any test prediction. If none passes, it exits with
status 2 without evaluating the test set or producing a deployment bundle.
Every model and the constant baselines are then evaluated on the same held-out
episodes, using the original paper metrics and episode-bootstrap confidence
intervals. These are engineering acceptance targets, not paper reproduction.

Use `--device cuda:0` to train the value on a working CUDA GPU. Collection and
the portable deployed value still run on CPU. The host's NVIDIA kernel and
userspace versions match as of October 4; the September overlay is unnecessary.

The continuation script accepts the qualified nominal actor, applies a randomized
ball-free walking gate, and then performs 1,000 avoidance updates in SPARK.
It can also wait for a running nominal trainer's checked export. It collects fresh
rollouts, trains/evaluates the new monitor, and checks the saved bundle online:

```bash
.venv/bin/python scripts/finish_spark_retraining.py \
  --spark-root /path/to/spark_dev \
  --spark-python /path/to/spark_dev/.venv-locomotion/bin/python \
  --nominal-run /path/to/spark_dev/outputs/locomotion/my_nominal_policy/actor \
  --output logs/safety_analysis/spark_29dof/my_continuation
```

Add `--avoidance-run /path/to/checked_avoidance_export` to use an already
trained actor. It must include the hash-checked `checkpoint.pt`, `policy.onnx`,
`deploy.yaml` and `result.json`. This skips PPO for avoidance and applies all
walking, physical-ball, robustness and monitor gates to that exact export.

Reserve the test cohort before selecting a candidate. `--qualification-seed N`
sets the quiet-test seed; physical-ball and disturbed Isaac tests use N+20 and
N+40. `--avoidance-episodes 128 --isaac-episodes-per-command 32` increases the
cohorts without relaxing any quality limit. If a held-out test fails, retain
that result and reserve a new cohort before further development; do not reuse
the exposed test as an independent final result.

Run it with the same working CUDA library environment as policy training. Its
atomic `status.json` records progress and failures. A failed nominal check may
trigger checked PPO resume within the explicit additional budget (default zero).
Keep that budget at zero for distilled actors: they support explicit weight
initialization for avoidance, but not checked PPO resume. For an ordinary PPO
nominal checkpoint, `--nominal-extra-iterations 2000 --nominal-chunk-iterations 1000`
allows additional training if needed.
It stops before avoidance/value training if that budget is exhausted. If avoidance loses walking
quality, it stops before monitor training. The progression gate uses eight
randomized 20-second resets per command in each simulator, zero falls, and
velocity RMSE at most 0.05 m/s standing and 0.2 m/s at ±0.3 m/s commands.
This is an engineering gate, not a paper criterion. Results and trajectories remain available;
the script never changes deployment defaults or commits files. `--check-only`
exercises just the gate on an already completed export.

Avoidance must also pass SPARK's paired physical-ball evaluation: at least 32
twenty-second trials at each of 0 and ±0.3 m/s, ≤5% falls, contacted-flight rate
at most half that of the same actor with masked ball observations, and ≤25%
contacted flights while standing. Physical balls remain present in both arms.
The exact exported actor must then survive at least 95% of 64 Isaac trials
at commands 0, ±0.3 and 0.6 m/s, with physical balls, material/mass
randomization, observation noise and planar pushes every five seconds.
The Isaac gate also requires robot self-collisions to be enabled, matching
the original Unitree task and SPARK's MuJoCo model.
The continuation then requires validation and final held-out monitor accuracy
before checking online execution in both MuJoCo and Isaac. These online checks
verify finite predictions; the separate held-out test measures accuracy.
A completed PPO run or finite value output alone does
not satisfy these gates.

Collection freezes the actor and deploy YAML inside the output directory.
The value configuration records their hashes and the dataset manifest hash.
Each validation-qualified and evaluated run produces `bundle.json`, which references the exact actor,
configuration, value checkpoint and held-out results. The monitor checks these
files and their association before starting. It refuses changed weights,
mismatched pairs and unrelated evaluation results. A bundle records an
evaluated pair; inspect the reported accuracy before drawing safety conclusions.
To run the final accuracy gate independently:

```bash
.venv/bin/python scripts/validation/qualify_spark_monitor.py \
  --bundle logs/safety_analysis/spark_29dof/my_spark_monitor/bundle.json \
  --output logs/safety_analysis/spark_29dof/my_spark_monitor/qualification.json
```

Stages are also available separately through
`scripts/validation/benchmark_spark_value.py collect|train|evaluate`. Collection
requires `--policy` and `--config`; it has no implicit released-actor fallback.
All output directories must be new; failed runs remain available for diagnosis.

## Run the learned monitor

```bash
.venv/bin/python safety_value/scripts/2_safety_analysis/run_monitor.py \
  --bundle logs/safety_analysis/spark_29dof/my_spark_monitor/bundle.json \
  --backend mujoco --viewer --velocity-command 0.3 0 0 \
  --output logs/safety_analysis/spark_29dof/my_viewer/trajectory.hdf5
```

The robot uses SPARK's built-in G1 scene. Policy and monitor run at 50 Hz of
simulation time, with 200 Hz physics. The observer does not change the policy's
actions. Add `--software-rendering` if native GLX is unavailable.

## Verify the old repositories are unnecessary

On Linux with `bwrap` installed, prefix MuJoCo collection, value training or
MuJoCo monitoring with:

```bash
bash scripts/validation/without_legacy_repos.sh .venv/bin/python ...
```

This runs the command without network access and hides both auxiliary Unitree
repositories, this repository's old Isaac extension and the released SPARK asset
cache in a private mount namespace. It does not rename or modify those folders.
SPARK's installed simulator and built-in model files remain available. Isaac
training can use the same hidden directories, but its default ground asset
comes from NVIDIA's asset service and currently requires network access.
