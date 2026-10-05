# SPARK retraining experiment, 2026-09-24

The SPARK-only pipeline has completed real Isaac policy training, MuJoCo rollout
collection, fresh λ-value training, and held-out evaluation. **The completed
avoidance actor did not pass locomotion qualification.** The resulting bundle is
an experimental policy/monitor pair, not a qualified replacement policy. The
September nominal continuation also finished and failed its walking gate; see
the [October training audit and qualified walking result](SPARK_TRAINING_PARITY.md).
The October work subsequently qualified new nominal and avoidance actors and
a matching monitor. The default viewer now uses that SPARK-trained pair;
the September actor and monitor below are historical. See the October report
for current checkpoints, accuracy, simulator compatibility and viewer commands.

## Policy training

All training weights in these runs originate from random initialization in
SPARK. No released actor or checkpoint from either auxiliary Unitree repository
was used to initialize them. A released actor was evaluated separately as a
diagnostic only.

The completed avoidance run is
`spark_dev/outputs/locomotion/avoid_seed42_20260924_v3`. It performed 1,000 PPO
iterations with 1,024 environments and 24 steps per iteration: 24,576,000 new
transitions. It started from our own 401-iteration SPARK balance checkpoint,
reset the optimizer and curriculum, and used 300 additional ball-free iterations
before 700 iterations with physical ball launches. Checkpoint metadata preserves
the parent hash and initialization settings. The exported ONNX actor agrees with
the Torch actor within 7.16e-7 maximum absolute error in the export checks.

Fresh MuJoCo checks used eight episodes per command, seeds 41000–41007, and a
20-second limit, with SPARK's built-in 29-DoF G1 and physical ball launches:

| Forward command (m/s) | Falls | Linear velocity RMSE (m/s) |
|---|---:|---:|
| 0.0 | 1/8 | 0.2337 |
| +0.3 | 2/8 | 0.3837 |
| −0.3 | 4/8 | 0.4615 |

Ball-free trials then removed the physical ball and supplied zero ball
observations, matching the training warm-up. Across four 20-second episodes per
command, the actor had no falls. Linear RMSE was 0.0074 m/s for standing,
0.2999 m/s for +0.3 m/s, and 0.2988 m/s for −0.3 m/s. It can stand quietly but
largely ignores walking commands. Tracking error during ball trials also
includes intentional evasion and must not be interpreted alone as walking error.
These ball-free checks use the same fixed nominal reset; their repetitions are
not independent samples of randomized initial conditions.

A separate 500-update locomotion-focused run learned steadier balance but still mostly stood: its
Isaac forward-command check had 0/3 falls and 0.2969 m/s RMSE for a +0.3 m/s
command. This behavior occurs in Isaac as well as MuJoCo.

Training investigation corrected the built-in URDF's contact grouping: its
eight fixed toe/heel links must count as feet, rather than unwanted body
contacts. Ball observation columns are now initialized to zero during a
ball-free warm-up, and command curriculum advances at episode-period intervals
using all episodes completed during each interval. Checking only resets exactly
on the boundary missed successful episodes with different reset phases.
The completed avoidance run predates the curriculum cadence correction; its
saved source diff records the implementation it used. Early failed experiments
remain under ignored output directories and are not deployment defaults.

## Monitor trained on the completed experimental actor

The [bundle](../logs/safety_analysis/spark_29dof/spark_scratch_candidate/bundle.json)
contains an exact, hash-checked actor/config/value pairing. Collection used 512
training, 128 validation and 256 test episodes, up to 1,000 control steps each.
Seed ranges are disjoint by whole episode. The λ model was initialized from
scratch and trained for 2,000 updates on the GPU with λ=δ=0.99, horizon 200, and
two 256/256 critics. Test data were not used for training or checkpoint selection.

The [held-out results](../logs/safety_analysis/spark_29dof/spark_scratch_candidate/evaluation_test/results.json)
cover 179,787 frames from 256 episodes, including 59,130 unsafe and 120,657 safe
frames. There are 1,037 ball segments, 1,007 labeled unsafe. Confidence intervals
resample whole episodes 1,000 times.

| Metric | Estimate | 95% episode-bootstrap interval |
|---|---:|---:|
| Temporal recall | 97.51% | 97.11–97.82% |
| Paper-defined false-safe rate | 10.17% | 8.30–12.38% |
| False-alarm rate | 2.29% | 1.95–2.72% |
| Invariant sign accuracy | 95.12% | 94.25–95.81% |
| Paper sample-value error | 0.1209 | 0.1011–0.1444 |

The always-unsafe baseline has 100% false alarms, and the always-safe baseline
has zero temporal recall. The actor's frequent unsafe encounters make recall
alone insufficient to assess the monitor. These are within-SPARK experimental
results; they do not establish paper reproduction or good locomotion.

Artifact identities:

```text
actor:    f493a2baa43f9c8c1bcfbe66404ffc824a5e31da65b263c0c6293bcb3924795a
deploy:   db5a84cfd59df250430b208fc7e7e72bec18a0d5ab961afea18185cc45d1fe65
dataset:  03aedf6cd3b97db6e9c0200d0bd0bd7a7e41225b43e370792c97ecff809912cd
value:    b7e5ccb9df8c625663f2e3c3162c36470694bbf30cfe31104ef84b17b6fa0ee4
```

## Independence and reproduction

Isaac training ran with both old Unitree repositories, the entire
`lambda_reach` repository, and the released SPARK asset cache hidden in a private
mount namespace. It used SPARK's built-in robot and the installed Isaac runtime.
NVIDIA's default ground asset still requires its asset service.

The complete rollout/value/evaluation command ran through
`scripts/validation/without_legacy_repos.sh`, with network disabled and both old
repositories, the old research Isaac extension, and the released asset cache
hidden. No folders were renamed or deleted. This validates code independence
for the exercised workflow, separately from policy quality.

Online verification used the saved bundle in that same offline namespace:
four fresh episodes, seed 50000 onward, +0.3 m/s forward command, and up to
1,000 steps. All 3,184 value predictions were finite. Two episodes ended in
falls, consistent with the actor's failed qualification. Headless throughput
was approximately 194 control steps per wall-clock second; simulation control
remained 50 Hz with 200 Hz physics. The trajectory and summary are saved beside
the bundle as `online_check.hdf5` and `online_check.summary.json`.

See [SPARK_RETRAINING.md](SPARK_RETRAINING.md) for training and bundle commands.
To inspect this experimental pair with the viewer:

```bash
cd /home/yifan/Research/Lambda/lambda_reach
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python safety_value/scripts/2_safety_analysis/run_monitor.py \
  --bundle logs/safety_analysis/spark_29dof/spark_scratch_candidate/bundle.json \
  --backend mujoco --viewer --software-rendering --no-download \
  --episodes 4 --steps 1000 --seed 50000 --velocity-command 0.3 0 0 \
  --output logs/safety_analysis/spark_29dof/scratch_candidate_viewer/trajectory.hdf5
```

All source changes remain staged for review. No new commit was made in either
repository. Generated checkpoints, datasets and simulator caches remain ignored.

## Completed September continuation

The final export completed 1,000 nominal-stage updates on September 24. Its
20-second ball-free checks measured standing RMSE .0123 m/s and forward RMSE
.2959 m/s. The backward trial fell after 2.96 seconds, with RMSE .5701 m/s.
`spark_final_continuation/status.json` records `nominal_quality_failed` and
`finished: true`; avoidance and monitor stages were not run by that invocation.
Both September services are inactive. The following is the historical launch
description, not a currently running job.

### Historical launch description

Nominal locomotion training continued from SPARK's own checkpoints, with
4,096 environments, no external velocity pushes or startup domain randomization,
and the corrected whole-period command curriculum. The target was 1,000 updates
in this nominal stage, including resumed updates. Its output is
`spark_dev/outputs/locomotion/avoid_seed42_20260924_nominal_final`.

The saved midpoint (501 nominal-stage updates) still fails the progression
gate: ball-free RMSE is 0.0055 m/s standing, 0.2972 m/s forward, and 0.2300 m/s
backward, with no falls in one 20-second nominal trial per command. Backward
tracking has improved, but forward tracking has not met the gate.

Two transient user services keep the work independent of the interactive chat:

```bash
systemctl --user status lambda-spark-locomotion-20260924.service \
  lambda-spark-continuation-20260924.service
cat logs/safety_analysis/spark_29dof/spark_final_continuation/status.json
```

The continuation waits for the completed export, checks nominal standing and
±0.3 m/s walking, then performs avoidance fine-tuning, fresh rollout/value
training, held-out evaluation, and online bundle verification. A failed initial
walking check stops that continuation and records `nominal_quality_failed`.
It does not change the default viewer. The gate was exercised on the completed
experimental actor above and correctly rejected it; the later stages have been
exercised individually. This September continuation subsequently stopped at the
failed nominal gate, as recorded above.

See [the continuation script](../scripts/finish_spark_retraining.py) and the
[live status](../logs/safety_analysis/spark_29dof/spark_final_continuation/status.json).
Completed service units are collected automatically; output logs and status
files remain available. Stopping the jobs, if desired, is explicit:

```bash
systemctl --user stop lambda-spark-continuation-20260924.service \
  lambda-spark-locomotion-20260924.service
```
