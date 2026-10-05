# SPARK training and monitor verification — 2026-10-05

**The SPARK-only 29-DoF workflow has passed its nominal walking, avoidance and
reachability-monitor gates.** This report records training lineage, held-out
policy/value results, online execution and WBT/SPORT/native SONIC compatibility.
The two legacy Unitree checkouts are archived outside the active project folder.
The SPARK implementation and pretrained actors are on SPARK `master` at
`fd8e4df`; the Lambda Reach integration is maintained on its `spark` branch.
Run directories referenced below remain local audit artifacts; monitor weights
and datasets are not included in Git.

The nominal recipe uses two fresh SPARK Isaac PPO teachers—one for standing,
one for walking with an observable gait clock—and supervised distillation into
one actor. Deployment runs that actor directly; it does not select between
teachers or substitute a scripted standing controller. No released weights
were used for training. Released actors were used only for diagnostic comparisons.

## Nominal actor result and validation

Final run: `/home/yifan/Research/spark_dev/outputs/locomotion/spark_g1_walk_20261005`.
It contains `policy.onnx`, `deploy.yaml`, `checkpoint.pt`, training lineage,
`qualification.json`, `command_transitions.json`, `gait_inspection.json`, and
`manifest.json`.

Policy SHA-256:
`f27744c41475a7fa746b117d234ee47d36dbc1ca5abd8ef39baddc60475feb04`.

Eight randomized episodes per command and backend, seed 97031:

| Backend | Standing RMSE (m/s) | +.3 m/s RMSE | −.3 m/s RMSE | Falls |
|---|---:|---:|---:|---:|
| SPARK MuJoCo | .0163 | .0762 | .0826 | 0/24 |
| SPARK Isaac | .0187 | .0868 | .0980 | 0/24 |

All episodes completed 1,000 control steps at 50 Hz. The engineering gate is
zero falls, standing RMSE ≤.05 m/s and walking RMSE ≤.2 m/s. These thresholds
are not paper accuracy claims. The user-selected built-in robot remains a
different plant from the original training USD and the paper's hand-equipped G1.

Eight additional 50-second MuJoCo trials, seed 98047, changed commands through
0 → +.3 → 0 → −.3 → 0 every ten seconds, with no falls. Braking takes time:
average RMSE over each full stop segment was .0733 m/s after forward walking
and .0551 after backward walking. Excluding the first two seconds gives
.0425 and .0138 m/s respectively. The result file retains every segment.

Kinematic replay of the held-out MuJoCo trials confirms alternating walking
steps. In all eight zero-command trials, after the first two seconds there
were no foot landings and the 1st–99th percentile foot-height range stayed
below .076 mm. This checks foot motion and geometric support, not contact forces.

Distillation observations came from seed 71000. Candidate screening used
96017; final qualification used 97031. The production distillation code
reproduced the exact same actor and config hashes with networking, both legacy
repositories, the entire research repository and the released cache hidden.
Torch/ONNX maximum absolute error was 9.54e-7. Qualification also hid legacy
code and released cache; Isaac retained network access for NVIDIA's ground asset.

## What the audit established

Source reference: `unitree_rl_lab-lambda` commit
`1402c9c01ef97644d08a698e94b0217b3c6b3805`, including the 29-DoF
`velocity_env_cfg.py`, `mdp/rewards.py`, PPO configuration, robot configuration
and training launcher. The historical `source/hj_humanoid` files in
`lambda_reach` supplied the paper-specific observation/avoidance contract.
These are read-only source references, not runtime dependencies.

| Area | Finding and change |
|---|---|
| Physics/control | Both use 5 ms physics and four substeps: 50 Hz control |
| Motors | SPARK locomotion now uses reference implicit drives, torque/speed limits, armature .01 and zero joint friction; generic backend defaults are preserved |
| Reset velocity | Unitree multiplies zero default joint velocities by (-1, 1); SPARK had sampled absolute velocities. Fresh training now resets to zero; legacy resumes preserve their old behavior |
| Push disabling | A [0, 0] push range brakes the robot periodically. `push_interval=0` now explicitly disables pushes |
| Curriculum | Reference boundary-reset sampling is selectable; old checkpoints retain SPARK's window averaging |
| PPO | Reference 512/256/128 ELU networks, 24 steps, five epochs, four minibatches, γ=.99, GAE=.95, std 1 and entropy .01 were audited and retained in the selected teachers |
| Warm starts | Learned per-joint exploration is preserved unless explicitly reset; fixed learning rates are available for refinement |
| Throughput | Cached static contact names and history scales; parked inactive balls without repeated teleports. A 200-step profile improved from 21.219 to 5.745 seconds, with an exactly matching seeded robot/reward trace |

All 20 checked reward terms matched independently executed Unitree and
IsaacLab functions on 64 synthetic states, with maximum error zero. The frozen
fixture and source hashes are in SPARK's `tests/fixtures/locomotion` directory.
A separate audit of 4,800 physical walking states found frame errors ≤5.41e-7;
joint state, acceleration, effort, limits and foot kinematics matched direct
articulation values exactly. Foot contacts detected alternating support.

A released reference actor survived 48 trials with SPARK's full training
perturbations (material/mass randomization, observation noise and pushes), with
standing/forward/backward RMSE .0989/.1085/.1326 m/s. This establishes that the
SPARK plant can execute robust walking. It does not prove that every recipe
will learn it, or that the new nominal actor has equivalent disturbance robustness.

Final automated checks passed: 52 SPARK locomotion tests and 17 research
integration, continuation and value-benchmark tests. They cover the reference
fixture, clock/reset/export contract, real optimizer updates, distillation
lineage and checkpoint handling, and workflow qualification gates. The new
one-command orchestrator is unit-tested; its individual training, distillation
and qualification stages were executed during this investigation, rather than
rerunning the entire expensive pipeline after packaging it.

No single exclusive cause of the original convergence failure was established.
The original-recipe runs learned standing within the tested budgets; the
released nominal training budget was approximately 1.18 billion transitions,
larger than these comparisons. The successful recipe exposes the gait phase
already used by the reward and explicitly combines learned stopping and walking.
Those are deliberate task/training adaptations, not claims of exact Unitree parity.

## Successful training lineage

1. **Standing teacher:** fresh 480-input actor / 495-input critic, 1,801 PPO
   updates × 4,096 environments × 24 steps. Reference disturbances and
   boundary curriculum, corrected resets, built-in SPARK G1.
2. **Walking teacher:** fresh 512-input actor / 527-input critic, 1,001 PPO
   updates at the same batch size. Adds sine/cosine of the .8-second gait phase,
   uses full straight-line commands in [-.5, 1] m/s, and initially disables
   pushes and startup randomization. All 29 actions remain learned.
3. **Distillation:** 6,000 Adam updates, batch 512, learning rate 1e-4, seed
   83051. Real SPARK MuJoCo observation histories from both teachers are used;
   commands and standing-clock values are augmented before querying the frozen
   SPARK teachers for action targets. The resulting actor handles both modes.

The selected teachers account for 275,447,808 PPO transitions. Distillation
updates are recorded separately and are not PPO iterations. The checkpoint's
critic is inherited; no PPO optimizer history is claimed for the distilled
stage. Subsequent PPO must use explicit weight initialization, not checked resume.

The walking teacher itself was verified in the actual Isaac training
environment: 32 held-out 20-second trials at 0, +.3, −.3 and +.6 m/s, seed
95017, with no falls. Mean walking speeds were +.275, −.310 and +.612 m/s.
Its remaining zero-command stepping motivated distillation.

Other bounded experiments did not supply weights to the final actor: earlier
quiet-task curricula remained near standing; a stronger tracking-reward warm
start initially destabilized; posture and leg-speed stopping penalties retained
walking but did not meet the standing gate. Their logs, recipes and source
snapshots remain under `spark_dev/outputs/locomotion/debug_20261005` for audit;
unsuccessful refinement recipes were removed from the maintained example.

## Reproduce and view

From SPARK, the complete workflow trains both teachers, checks their required
behavior, distills one actor, and runs held-out qualification:

```bash
cd /home/yifan/Research/spark_dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-locomotion/bin/python example/unitree_g1/locomotion/train_walking.py \
  --output outputs/locomotion/my_walking_run
```

A failed teacher or final gate stops the workflow and records `status.json`.
The default iteration counts reproduce the selected recipe; they are not a
convergence guarantee across hardware or software versions. The working local
stack is Python 3.12, Isaac Sim 6.0.0.1, IsaacLab 13 development, Torch 2.11,
RSL-RL 3.0.1 and matching NVIDIA 580.178.04 kernel/userspace libraries. Do not
source the historical September driver overlay on this host.

View the qualified actor on the built-in SPARK MuJoCo scene:

```bash
.venv-locomotion/bin/python example/unitree_g1/locomotion/run_benchmark.py \
  --backend mujoco --family avoid --viewer --no-ball --no-download \
  --policy outputs/locomotion/spark_g1_walk_20261005/policy.onnx \
  --config outputs/locomotion/spark_g1_walk_20261005/deploy.yaml \
  --velocity-command 0.3 0 0 --episodes 2 --steps 1000 \
  --output outputs/locomotion/walking_viewer.hdf5
```

Use a fresh output filename on repeated runs. `--backend isaac` selects Isaac.
The `avoid` family here identifies the ball-compatible input layout; this actor
has only nominal locomotion training.

## Qualified avoidance and reachability monitor

The target remains SPARK plus `lambda_reach`. Both nominal walking and the new
avoidance actor have now passed their policy gates. The qualified avoidance
package is `spark_dev/outputs/locomotion/spark_g1_avoid_20261005`, actor SHA-256
`90735dd9f346ab99365ffbaf047bde219ad99df8129b2e3e976dbc21408eca54`.
Its retained training lineage starts with the SPARK teachers above and contains
290,537,472 additional avoidance transitions. Released actors supplied no weights.

The frozen actor passed 48 quiet standing/±.3 m/s walking trials across both
engines without falling. In 384 held-out MuJoCo ball trials it fell three times,
contacted 19.91% of standing flights, and reduced overall contacts by 69.45%
relative to paired trials with its ball observations masked. It fell once in
128 held-out Isaac trials with physical balls, full material/mass randomization,
observation noise and periodic pushes. Every declared policy gate passed.
An earlier candidate failed the standing-contact limit; its result is retained,
and the selected actor was tested on a new cohort reserved before selection.

Fresh collection for the exact actor used 1,638 training, 410 validation and
512 test episodes. Each episode uses a constant forward command sampled from
[-.5, 1] m/s, random initial XY/yaw and physical ball launches in SPARK MuJoCo.
The test rollouts had 10 falls in 512 episodes. Descriptive command bins had
1/64 falls at [-.5, -.3), 0/109 at [-.3, 0), 0/86 at [0, .3), 3/120 at [.3, .6),
and 6/133 at [.6, 1] m/s; these bins were not additional selection gates.
This is the monitor's evaluated
distribution; disturbed Isaac accuracy is not inferred from MuJoCo accuracy.

Three λ seeds and three supervised baselines were trained from random weights
for 2,000 updates each. λ uses two 256/256 critics, λ=δ=.99 and horizon 200.
Seed 2 was selected by validation gates and lowest validation sample-value
error. `selection.json` froze its checkpoint before any test prediction.
All three λ seeds passed the final gates. For the selected model, 512 test
episodes supplied 508,145 scored frames and 1,076 unsafe segments:

| Metric | Estimate | 95% episode-bootstrap interval | Gate |
| --- | ---: | ---: | ---: |
| Temporal recall | 97.76% | 97.68–97.86% | ≥95% |
| Paper-defined false-safe rate | 3.69% | 2.77–4.80% | ≤10% |
| False-alarm rate | 11.00% | 10.52–11.43% | ≤15% |
| Sample-value error | .1347 | .1245–.1455 | ≤.20 |

Supervised baselines performed similarly on this distribution (sample-value
errors .1352–.1508). The always-safe baseline has 90.09% invariant accuracy
because most frames are safe, but misses every unsafe frame. The selected
monitor's 89.73% invariant accuracy therefore should not replace the separate
miss/false-alarm and temporal-recall measures above. These results establish
within-SPARK monitor quality, rather than exact paper reproduction or an
algorithm-superiority result.

The final bundle is
`logs/safety_analysis/spark_29dof/spark_replacement_20261005_v2/monitor/bundle.json`.
Value SHA-256:
`7659dc77f137fffbd2178fc42401cdc9fde984083a0df0ac721121e9b06f5007`.
Dataset SHA-256:
`c265f9c8bba93145efb9acf63ccaef3bd02dc70a03b6da3e1e813cd7580d11a5`.
The bundle binds exact policy/config/value bytes to the held-out result.
Online execution produced 4,000 finite predictions and zero falls in each
backend's four 20-second trials, with matching policy/value hashes. Mean
headless throughput was 223.5 control steps per wall second in MuJoCo and
32.6 in scalar Isaac deployment. Both simulations use 50 Hz control and
200 Hz physics; Isaac's scalar monitor deployment ran slower than real time.

The default viewer now loads this bundle. A live software-rendered launcher
check produced 100 finite predictions, no fall and 49.9 wall-clock control Hz
with a 20 FPS display. From `lambda_reach`:

```bash
bash deploy/avoid/run_spark_monitor.sh --software-rendering \
  --velocity-command 0.3 0 0
```

Training/qualification hid both legacy repositories and the default released
asset cache. MuJoCo collection/value/monitor runs also disabled networking.
After qualification, the clean legacy checkouts were moved intact to
`/home/yifan/Research/Lambda_legacy_archive_20261005/`, with their Git histories
and ignored files retained. The active Lambda folder now contains
`lambda_reach`; SPARK remains `/home/yifan/Research/spark_dev`. A further
offline monitor run passed with the old paths absent. The archive manifest
records original paths and commits for restoration.

The avoidance follow-up found and corrected disabled robot self-collisions
in the locomotion setup; the original Unitree task and SPARK MuJoCo enable
them. The nominal actor was then rechecked with self-collisions enabled in
Isaac, using eight 20-second episodes each at 0 and ±0.3 m/s, seed 120011.
It had zero falls in all 24 trials, with RMSE 0.0173, 0.0894 and 0.0917 m/s.
Its weights are unchanged. The canonical manifest references the additional
`qualification_self_collision_isaac.json` report, including plant settings
verified against each trajectory's metadata. The follow-up avoidance and
monitor qualification is documented in SPARK's locomotion `VALIDATION.md`.
That report also records real WBT/SPORT Isaac runs with active base and arms
and 32 stable native SONIC execution probes. Their measured trajectories match
the pre-migration backend. The separate SONIC C++ server remains unavailable
because its external executable/assets are absent.
