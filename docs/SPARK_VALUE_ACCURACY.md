# Value accuracy on SPARK's built-in G1

This is the historical September released-policy experiment. The current
viewer uses the newly SPARK-trained actor and monitor qualified in the
[October training and monitor report](SPARK_TRAINING_PARITY.md).

The acceptance target is now predictive accuracy on SPARK rollouts. Numerical
agreement with the original MuJoCo trajectories is not required. The user chose
SPARK's built-in 29-DoF G1 on 2026-09-23. The current experiment concerns the
released **avoidance** actor and a value trained from scratch for that actor.

**Measured verdict:** warning performance is close to the supervised SPARK
baseline and substantially better than DPE. The current λ configuration does
not yet match the paper across all metrics: its value error is higher, and it
raises warnings on about 15% of observed-future safe frames. This evaluation
supports SPARK as a useful value-learning environment; it does not establish
that the current λ value is fully qualified for reliable monitoring.

## Reference and interpretation

[The paper, Table 1 and Appendix C](https://arxiv.org/html/2606.16022)
reports these G1Collision simulation results:

| Method | Temporal recall (%) | Sample value MSE | False-safe rate (%) |
| --- | --- | --- | --- |
| λ-reachability, λ=0.99 | 99.40 ± 4.79 | 0.13 ± 0.23 | 6.12 |
| Supervised | 99.72 ± 2.60 | 0.14 ± 0.27 | 2.05 |
| DPE | 22.61 ± 14.50 | 0.18 ± 0.37 | 60.47 |

The paper calls the last column “false-positive rate,” with **safe as the
positive class**: `P(V <= 0 | future maximum safety signal > 0)`. It measures
missed hazards, not false alarms. This benchmark reports both directions.
Temporal recall measures warning lead time divided by the available time from
the disturbance to the first violation. Value error uses the observed future
maximum within each disturbance segment.

Those published numbers are reference targets. The paper's simulation robot,
policy inputs and simulator differ from the current 29-DoF SPARK system.
Matching the numbers would demonstrate similar measured performance on this
new benchmark, not exact paper reproduction or hardware safety certification.

## Frozen experiment

- Built-in `UnitreeG1WholeBodyDynamic1Config`, MuJoCo, 5 ms physics / 20 ms policy.
- Released 29-DoF avoidance actor; no actor training or checkpoint changes.
- 1,638 training episodes, 410 validation episodes, 512 independent test
  episodes; up to 1,000 steps (20 seconds) each. Seeds are disjoint ranges.
- Each episode samples initial xy and yaw and a constant forward command in
  `[-0.5, 1.0] m/s`; lateral and yaw commands are zero. SPARK supplies the ball
  launches (2–5 m, 3–6 m/s, every 3–5 seconds), dynamics and termination.
- No extra mass/friction randomization or artificial large pushes. This tests
  the nominal built-in scene under the sampled commands and ball disturbances.
- Labels use the current monitor's `simulation_avoid` profile: tilt, torso
  contact, height clearance and latched ball contact. Configuration is saved
  verbatim in the dataset manifest. These are physical simulation labels,
  not the paper's hardware distance proxy.
- Retain the existing recorder convention: policy input at step t paired with
  the post-action signal. Segmentation uses the current adapter's post-step
  event flags. No future labels enter network inputs.
- Split whole episodes **before** balancing; downsample only training anchors
  to equal safe/unsafe counts, with fixed seed 42. Validation/test retain their
  natural prevalence. Segment endpoints use the paper's legacy terminal rule.
- All three methods use width `[256,256]`, ReLU, Adam 0.001, batch 256 and
  2,000 gradient updates. λ uses two critics, λ=δ=0.99, max horizon 200,
  regression only, target τ=0.05 every 10 steps. DPE uses the original
  convergence-triggered 0.9→0.99 schedule. All original trainers are reused.
  The supervised trainer retains its repository-default initialization; λ and
  DPE initialize their output heads to -2. DPE remained at 0.9 in these runs.
- Training seeds 0, 1, 2; final fixed-budget checkpoints, no test-based model
  selection. Validation curves are retained to assess convergence.
- Evaluate every method on the same test episodes with the original paper
  metric functions. Add false-alarm rate and always-safe/always-unsafe controls.
  Confidence intervals resample whole episodes, not correlated frames.

This is a substantial nominal-scene evaluation. It does not cover rough
terrain, push-task values, sensor noise, hardware transfer or all possible
disturbance distributions. Full data quantity alone does not establish
convergence; learning curves and held-out baseline gaps must also be inspected.

## Reproduction

From `lambda_reach`, use its existing `.venv` and new output directories. CPU
execution is supported; CUDA was unavailable on the verification machine due
to a driver/library mismatch. No driver or global environment changes are made.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
RUN=logs/safety_analysis/spark_29dof/value_accuracy/recheck
$PY scripts/validation/benchmark_spark_value.py collect --output "$RUN/data" \
  --policy "$HOME/.cache/spark/g1-locomotion-v1/avoid/policy.onnx" \
  --config ../../spark_dev/module/spark_policy/spark_policy/control/whole_body/unitree_g1/locomotion/resources/configs/avoid.yaml

for method in lambda_reachability supervised dpe; do
  for seed in 0 1 2; do
    $PY scripts/validation/benchmark_spark_value.py train \
      --data "$RUN/data" --method "$method" --seed "$seed" \
      --output "$RUN/${method}_seed${seed}"
  done
done

$PY scripts/validation/benchmark_spark_value.py evaluate \
  --data "$RUN/data" --models "$RUN"/*_seed* --output "$RUN/test"
$PY scripts/validation/report_spark_value.py \
  --data "$RUN/data" --evaluation "$RUN/test"
```

Collection and training refuse to overwrite existing output directories.
Dataset files have individual hashes; evaluation checks the training dataset
manifest and actor hash. Artifacts stay under ignored `logs/`; the maintained
code, tests and documentation stay in `lambda_reach`. The current viewer's
checkpoint is not automatically replaced by experimental checkpoints.

## Executed results — 2026-09-23

All 2,560 episodes completed: 2,048,000 training/validation frames and 512,000
test frames. All nine training runs completed their 2,000 updates. The test
set has 2,803 event segments, including 323 unsafe segments and 12,447 frames
with a positive observed future maximum. The original evaluator omits two
one-frame event tails, so 511,998 frames are scored. No robot falls occurred;
this experiment primarily qualifies ball-collision prediction, not recovery
from large destabilizing pushes.

| Method | Temporal recall (%) | Value MSE | False-safe (%) | False-alarm (%) | Sign accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| λ-reachability | 95.853 ± 0.071 | 0.201 ± 0.019 | 3.310 ± 0.229 | 14.970 ± 1.672 | 85.313 ± 1.626 |
| Supervised | 95.509 ± 0.009 | 0.110 ± 0.006 | 3.382 ± 0.197 | 12.017 ± 0.782 | 88.193 ± 0.758 |
| DPE | 28.488 ± 1.050 | 0.179 ± 0.003 | 48.395 ± 0.631 | 5.159 ± 0.234 | 93.790 ± 0.216 |

These are means ± sample standard deviations across three training seeds.
They are not the same uncertainty quantity as the paper table's reported
dispersion. Each individual run also has 95% confidence intervals from 1,000
whole-episode bootstrap resamples in the machine-readable results.

All λ seeds warned before the first violation in all 323 unsafe segments, but
that event-level detection rate does not imply every preceding frame was
classified correctly. Their average temporal recall is 3.55 percentage points
below the paper's λ result, while their false-safe rate is 2.81 points lower.
Their value MSE is about 55% above the paper's 0.13 and about 83% above the
supervised SPARK baseline. The numerical safety margin therefore remains a
meaningful gap, even though warning performance is strong.

λ validation MSE improves initially and then increases toward the end of
training. Supervised validation MSE stays lower. This indicates that simply
running more of the same λ updates cannot be assumed to solve the gap; target
estimation, balancing and calibration need validation-based investigation.
These results do not isolate the cause. Any subsequent tuned configuration
should receive a fresh final test cohort.

An always-safe predictor scores **97.57% sign accuracy** here while missing
every unsafe frame. An always-unsafe predictor gets perfect temporal recall
and zero false-safe errors while falsely warning on every safe frame. Those
controls demonstrate why no single accuracy/recall statistic is sufficient.

Artifacts (local, ignored by git):

- [Detailed report](../logs/safety_analysis/spark_29dof/value_accuracy/test/REPORT.md)
- [Per-model metrics and confidence intervals](../logs/safety_analysis/spark_29dof/value_accuracy/test/results.json)
- [Reference comparison plot](../logs/safety_analysis/spark_29dof/value_accuracy/test/comparison.png)
- [Validation curves](../logs/safety_analysis/spark_29dof/value_accuracy/test/learning_curves.png)
- [Dataset manifest](../logs/safety_analysis/spark_29dof/value_accuracy/main/manifest.json)
- [Exact training commands](../logs/safety_analysis/spark_29dof/value_accuracy/training_commands.json)

The newly trained models are under `logs/safety_analysis/spark_29dof/value_accuracy/`
in `lambda_reachability_seed{0,1,2}/models/last_seed_{0,1,2}.pt`, with matching
training configs. They are separate from the old 200-update smoke checkpoint.
The subsequent online-verification request updates `deploy/avoid/run_spark_monitor.sh`
to use `lambda_reachability_seed0`. This is the fixed seed-0 model, not a
checkpoint chosen for the best test score.

Metric-direction, event-boundary, episode-split, balancing and data-integrity
regression checks pass (`tests/test_spark_value_benchmark.py`). The benchmark
CLI was exercised end-to-end before the main experiment. Code and documentation
are staged for review; no commits or production simulator changes were made.
