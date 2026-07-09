# Real-Data Safety-Value Evaluation — Plan & Progress

**Status:** PLANNING / IN PROGRESS
**Owner:** ruic (richen@umich.edu)
**Created:** 2026-05-25
**Branch:** `deploy`

This is the hand-off document for the task of training all safety-value baselines/ablations on the
29-DoF G1 sim safety tasks and evaluating them on the real-robot experiment data. New chats should
read this top-to-bottom, then continue from the **Progress Tracker**.

---

## 1. Goal (restated)

We have two 29-DoF G1 sim "safety analysis tasks":
- **flat safety task** = `g1_29dof_flat_unitree_ppo_6000` (flat push / balance). obs_dim = **480**.
- **avoid safety task** = `g1_collision_avoid_fwd_back_4999` (collision avoidance). obs_dim = **510**
  (480 base obs + 30 = ball_pos history 3×5 + ball_vel history 3×5).

For each task, only the proposed method **λ-reachability @ λ=0.99** is trained so far
(`results/lambda_reachability_lambda_reach_lambda_0_99/`). We want to:

1. **Train all other baselines + ablations** on the SIM task data (train pkl), same as λ=0.99.
2. **Convert the real-robot data** (`logs/real_exp/...`) into the SIM data format, synthesizing two
   NEW safety-analysis tasks: **`real_push`** and **`real_avoid`** (all real data is test-only).
3. **Evaluate all trained safety values on the real data**, reproducing the full metric suite that
   the sim "inference evaluation" produces (temporal recall, detection rate, t_lead/t_unsafe,
   sample value error, invariant confusion matrix → acc/prec/rec/F1/FPR/FNR), plus the offline
   `test.py` metrics (value MSE, classification).

Real inference via *new* Isaac-Sim rollouts is **impossible** (only offline real logs exist). The
key realization: the sim "inference evaluation" is itself **fully offline** — it only needs, per
episode, `safety_signal` (ground truth), `safety_value` (model prediction), and `event` flags (for
segmentation). So we can reproduce all sim-inference metrics on real data without a simulator.

---

## 2. Codebase map (verified)

Safety-analysis framework lives in `safety_value/`. Pipeline stages under `safety_value/scripts/`:
- `0_policy/` — train/play PPO policies (not needed here).
- `1_data_prep/rollout_safety_data.py` — Isaac-Sim rollout → `data_raw.hdf5` (needs sim).
- `1_data_prep/data_processing.py` — `data_raw.hdf5` → `data_processed_{train,test}.pkl` (offline).
- `2_safety_analysis/train.py` — train a safety value (offline; torch).
- `2_safety_analysis/test.py` — eval on `data_processed_test.pkl` (offline; torch). **Accepts `--dataset_path`.**
- `2_safety_analysis/inference.py` — Isaac-Sim rollout + model → `inference/seed_X/dataset.hdf5` (needs sim).
- `3_result_analysis/evaluate_inference.py` — reads `inference/seed_X/dataset.hdf5`, computes metrics (offline).
- `run_pipeline.py` — orchestrates train→test→plot_train→inference→evaluate→report_table.
- `config/pipeline_algorithms.json` — algorithm/ablation definitions.

Algorithm families (`safety_value/safety_analysis_algos/`): `dpe`, `supervised`, `weakly_supervised`,
`lambda_reachability`. Models are MLPs via `build_mlp`; λ-reachability uses a 2-critic ensemble
(`critic1`,`critic2`) whose value = `0.5*(critic1+critic2)`.

### Data formats

**Raw HDF5 (`data_raw.hdf5`)** — group `data/`, one subgroup per episode (`demo_0`, `demo_1`, …).
Per-episode keys (from recorder `key_prefix`s):
- `policy_obs` `[T, obs_dim]` — observations (network input).
- `safety_signal_balance` (flat) / `safety_signal_collision` (avoid) `[T]` — ground-truth signal; **>0 = unsafe**.
- `event_push` (flat) / `event_ball_spawn` (avoid) `[T]` — 1 at the step a disturbance is triggered.
- `terminal_state` `[T]` — loaded but **not used** in processing (can be zeros).
- `stability_signal_vel_track` `[T]` — optional, viz only.

`data_processing.find_signal_keys` matches by prefix: exactly one `safety_signal_*`, one `event_*`,
plus `terminal_state` and `policy_obs` (required); `stability_signal_*` optional.

**Processed pkl** — `list` of 7-tuples
`(x, x_next, safety_signal, sample_safety_value, invariant_flag, x_future, safety_future)`:
- Episodes are split into **segments at `event_*` indices** (`np.split` at `event>0`).
- Per segment: `sample_safety_value[t] = max(safety_signal[t:end])` (unbounded future max);
  `invariant_flag[t] = (sample_safety_value[t] <= 0)`.
- Dataset is **class-balanced** (downsample majority of invariant vs not) then split train/test
  (`test_proportion`, default 0.2). `HJValueDataset` loads the whole pkl into RAM.

**Inference HDF5 (`inference/seed_X/dataset.hdf5`)** — same per-episode layout as raw **plus**
`safety_value` `[T]` (the model's prediction along the rollout). This is what
`evaluate_inference.py` consumes. `find_signal_keys` there needs `safety_signal_*`, `safety_value`,
optional `event_*`, `stability_signal_*`.

### Metrics

`test.py` (on `data_processed_test.pkl`, per sample): predict `v=model(x)`, `pred=(v<=0)`.
Confusion matrix vs `invariant_flag` → accuracy/precision/recall/F1/FPR/FNR, plus `inv_acc` and
`value_mse = mean((v - sample_safety_value)^2)`. Writes `results/<subdir>/test/results_step_*.json`.

`evaluate_inference.py` (on `inference/seed_X/dataset.hdf5`, per segment):
- `t_unsafe` = steps from segment start to first `safety_signal>0`.
- `t_lead` = steps from first `safety_value>0` (before the unsafe step) to first `safety_signal>0`.
- `temporal_recall = t_lead/t_unsafe`; aggregated mean/std/min/max/median over unsafe segments.
- `detection_rate` = fraction of unsafe segments with `t_lead>0`.
- `sample_value_error` = MSE(`safety_value`, future-max `safety_signal`) per segment, aggregated.
- invariant confusion matrix (per step, segment-aware future max) → acc/prec/rec/F1/FPR/FNR.
Writes `results/_inference_comparison/seed_X/inference_evaluation_seed_X.csv` + plots.

---

## 3. The real data (verified)

Location after copy: `logs/real_exp/`. Each demo dir has a timestamped subdir with
`records.csv` (+ usually `records.npy`), `context_raw.csv`, `metadata.yaml`.

- **Push demos (→ real_push):** `demo_push_1..4`, `push_1_front..push_4_front`,
  `push_5_left..push_8_left` = **12 demos**. obs_0..479 (480-dim). `metadata.safety_signal: tilt_only`.
- **Avoid demos (→ real_avoid):** `demo_avoid_1`, `demo_avoid_2` = **2 demos**. obs_0..509 (510-dim).
  (avoid `metadata.obs_dim:480` is **stale/wrong**; the CSV actually has 510 obs cols.)
- Other dirs (`0510`, `test_without_support`, top-level `*_velocity`) — ignore unless asked.

`records.csv` columns (per row = one 0.02s control step): `time_s, relative_time_s, control_step,
dt_actual, root_ang_vel_0..2, root_quat_0..3, projected_gravity_0..2, command_0..2, joint_pos_0..28,
joint_vel_0..28, prev_action_0..28, policy_action_0..28, target_q_0..28, obs_key, obs_0..N,
safety_valid, safety_value, safety_signal, safety_inference_ms, policy_inference_ms`.
- `obs_0..N` = the exact network input (N=479 push / 509 avoid).
- `safety_value` = the **deployed** model's logged prediction (≈ λ=0.99; validated to ~1e-7 by
  `scripts/real_exp_offline_inference.py`). NOT ground truth.
- `safety_signal` = the deployed **tilt_only** ground-truth signal.
- `records.npy` (when present) = structured array; field `numeric_data [T, 647or677]`,
  `numeric_columns`, `string_data` (obs_key). Faster than CSV; some CSVs have a few `dropped_bad_rows`.
- **No event/disturbance column anywhere.** ← the core missing piece.

### Ground-truth signal comparability (IMPORTANT)

- **Sim flat** `safety_signal_balance = max(tilt, base_contact, support_clearance)` where
  `tilt = (atan2(√(gx²+gy²), -gz) - π/4)/(π/4)` from `projected_gravity_b`.
  **Real push** records only the tilt term (`tilt_only`). Since a real fall makes tilt, base
  contact, and clearance all fire together, tilt-only is a **reasonable proxy**. Real push
  `safety_signal` was positive in only **4 / 30022** steps in demo_push_1 (robot almost always
  recovers) → very imbalanced, few unsafe segments.
- **Sim avoid** `safety_signal_collision`: distance `d=‖ball−robot‖`,
  `l = clamp((0.5−d)/0.5, −1, −1e-6)` normally, **`l=+1` on ball contact** (latched 0.25s); then
  `max` with tilt/body-contact/support. **Real avoid records only tilt_only**, which was **0 / 20074**
  positive (robot dodges/gets hit but never falls). ⇒ **Real avoid as-recorded has NO usable
  unsafety ground truth.** To evaluate avoid models meaningfully we must **reconstruct** a
  collision/proximity signal from the ball position embedded in the real obs
  (current ball_pos_base ≈ obs_492..494; current ball_vel_base ≈ obs_507..509, newest frame last),
  mirroring the sim formula (danger_distance=0.5; choose a contact-distance threshold since there is
  no real contact sensor). **This is a modeling decision — see Open Decisions.**

---

## 4. Environment (RESOLVED)

**Use the `~/IsaacLab/hj` venv, from bash (NOT zsh — repo doesn't work in zsh).** It already has
everything: Python 3.11, torch 2.7.0+cu128 (CUDA available), h5py 3.14, tqdm 4.67, pandas 2.3,
numpy 1.26, matplotlib 3.10, isaaclab 0.46, rsl_rl, safety-value 0.1.0. **No installs needed.**

Canonical invocation for every script (the assistant's Bash tool runs zsh, so wrap in `bash -c`):
```bash
bash -c 'source /home/ruic/IsaacLab/hj/bin/activate && cd /home/ruic/hj_humanoid && python <script> ...'
```
Offline scripts (train/test/data_processing/evaluate_inference + new converters) do not need
Isaac Sim, but this venv has it anyway. GPU present — use for value-net training/inference.

**Memory risk:** flat train pkl = **21 GB**, test = 5.3 GB; avoid train = **24 GB**, test = 5.9 GB.
`train.py` loads BOTH train+test into RAM via `HJValueDataset`. System has **31 GB RAM + 65 GB swap**.
λ=0.99 trained OK, so it fits (tight, likely uses swap → slow load). Train methods **sequentially**,
monitor `free -h`. Real test pkls are tiny by comparison.

---

## 5. Plan

### Phase 0 — Setup
- [ ] Copy `…/shared/logs/` → `logs/` (rsync, in progress).
- [ ] Install `h5py tqdm pandas` into `~/isaacsim/python.sh`.
- [ ] Smoke test: load each sim task's `training_config.json`; run `test.py` for the existing
      λ=0.99 model on its own sim test pkl to confirm the toolchain runs end-to-end.

### Phase 1 — Train all baselines/ablations on SIM data (per task)
Replicate the existing λ=0.99 config (`results/.../training_config.json`:
hidden_dims=[256,256], batch=256, lr=1e-3, total_steps=10000, epochs=20, eval_steps=500, seed=0;
λ params weight_hinge_lb/mono/bce=1.0, target_tau=0.005, target_update_period=1), changing only what
each method requires. Use `train.py` directly (NOT run_pipeline, which has different defaults), so
naming and hyper-params match.
- Method set + naming: **see Open Decision #2.** Candidate set (from `label_map.json`):
  `dpe`, `supervised`, `w_supervised` (weakly_supervised), λ-ablations `lambda_reach_lambda_0_95 /
  _0_50 / _0_00` (and maybe `_0_90`), and possibly loss-term ablations
  `lambda_reach_lb / lb_mono / lb_mono_bce`.
- Output goes to `logs/safety_analysis/<task>/results/<model_subdir>/{models,train,training_config.json}`.
- After each: run `test.py` on the SIM test pkl (sanity / sim numbers).

### Phase 2 — Build the real safety tasks (`real_push`, `real_avoid`)
New offline converter script (to be written, e.g. `safety_value/scripts/1_data_prep/real_to_raw.py`):
1. Load each real demo (`records.npy` if present else robust CSV parse; drop bad rows).
2. Extract `policy_obs = obs_0..N` `[T, obs_dim]`.
3. Ground-truth `safety_signal`:
   - push: use recorded `safety_signal` (tilt_only) — optionally recompute from `projected_gravity`
     to confirm identity.
   - avoid: **reconstruct** collision/proximity signal from ball obs (Open Decision #1).
4. Synthesize `event_*` flags (segment boundaries) — Open Decision #3. Persist a sanity plot per demo
   (safety_signal + synthesized events) for human review.
5. `terminal_state = zeros`; carry obs as `policy_obs`.
6. Write `logs/safety_analysis/real_push/data_raw.hdf5` and `…/real_avoid/data_raw.hdf5` with one
   group per demo, named so the demo identity is preserved (e.g. `demo_push_1`, `push_5_left`).
7. `data_processing.py --safety_analysis_root real_push --test_proportion 1.0` (all → test pkl). Same for avoid.
   (Confirm `test_proportion=1.0` yields empty train / full test cleanly; else patch.)

### Phase 3 — Evaluate all models on the real tasks
For each real task and each trained method (models come from the SIM task):
- **Route A (test.py):** place/symlink the SIM-trained `results/<subdir>/{models,training_config.json}`
  under `logs/safety_analysis/<real_task>/results/<subdir>/`, then
  `test.py --safety_analysis_root <real_task> --model_subdir <subdir>` (reads real test pkl).
  → classification + value_mse on real.
- **Route B (evaluate_inference.py):** new offline filler script (generalize
  `scripts/real_exp_offline_inference.py`): for each demo, run the method's model on `policy_obs`
  → `safety_value [T]`; write `logs/safety_analysis/<real_task>/results/<subdir>/inference/seed_0/dataset.hdf5`
  containing per-demo `safety_signal_*`, `event_*`, `safety_value` (+ optional `stability_signal_*`).
  Then `evaluate_inference.py --sa_roots <real_task> --methods <all subdirs> --seed 0` → full
  inference metric suite (temporal recall, det rate, SVE, invariant CM) + comparison plots/CSV.
- Cross-check: for λ=0.99 on real, the Route-B `safety_value` should match the logged `safety_value`
  column to ~1e-6 (validates obs/model wiring).

### Phase 4 — Report
- Aggregate real metrics into a table per task (mirror `report_table` / `_inference_comparison` CSV).
- Sim-vs-real comparison table for the same methods.

---

## 6. Decisions (RESOLVED by user 2026-05-25)

1. **Avoid ground-truth signal.** Online it was `max(tilt, contact)` but the recorded data appears to
   have only the tilt component (contact not recorded / change not pushed). **Reconstruct contact**
   from the ball position in the real obs: if ball–humanoid (torso) distance ≤ **0.30 m** →
   `contact = +1`, else `contact = −1`. Real avoid `safety_signal = max(tilt_signal, contact)`.
   (Binary contact, per user — not the sim's graded `(0.5−d)/0.5` distance signal.)
2. **Method set.** In addition to existing **λ=0.99**, train: **λ=0.95, λ=0.50, λ=0.00, dpe,
   supervised**. (NO weakly_supervised, NO loss-term ablations.) Naming: match the existing dir
   convention — λ family uses `--model_tag lambda_reachability --run_name lambda_reach_lambda_0_XX`
   → `lambda_reachability_lambda_reach_lambda_0_95/_0_50/_0_00`; `dpe` and `supervised` use default
   tag → subdirs `dpe`, `supervised`. (Add label_map entries later for pretty plot labels.)
3. **Event synthesis = auto-detect + whole-data sanity plots for user approval/tuning.**
   - **Push:** detect push onsets from spikes (root_ang_vel / projected_gravity). Contiguous
     segmentation (every step in a segment), like sim.
   - **Avoid:** **ball falling to the ground = robust END-of-event** marker. For each landing, the
     segment is the **throw**: backtrace from landing to **throw onset** (ball starts flying). The
     **"prep" period** (grab ball, walk around, then finally throw) BEFORE throw onset is
     **excluded** — unlike sim, real segments are **non-contiguous** (gaps between them). Implement by
     extracting each throw segment as its own HDF5 episode and dropping prep periods.
   - Sanity plots must show the **whole** demo with prep / event-onset / segment / landing marked,
     for the user to visually approve and tune thresholds before finalizing.
4. **Environment = `~/IsaacLab/hj` venv via bash** (see §4). All packages present; no installs.

---

## 7. Risks / gotchas
- 21–24 GB train pkls vs 31 GB RAM → train methods sequentially; watch swap; loads are slow.
- Naming mismatch (Decision #2) will break `evaluate_inference` label lookup if inconsistent.
- Real avoid has only 2 demos and (as recorded) no positives → metrics fragile; reconstruction matters.
- Real push positives are extremely sparse (≈4/30k) → temporal recall computed on very few segments;
  FPR / value calibration may be the more meaningful real metrics.
- `terminal_state` unused in processing (zeros OK). `data_processing` balances classes — for a
  test-only real set we want `test_proportion=1.0`; verify balancing doesn't drop the rare positives
  in a harmful way (may want to disable balancing for the real/test conversion).
- obs history ordering (newest = last frame) assumed from `real_exp_offline_inference.md`; verify the
  ball-obs index slice before trusting the reconstructed avoid signal.

---

## 8. Progress Tracker

**Workflow (per user, 2026-05-25):** interactive — assistant proposes the next step + exact command;
user runs it; both inspect outputs; iterate. Heavy/training commands are run by the user (in bash).

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0.1 | Copy shared/logs → logs/ | DONE | dest=src=70 GB; rsync exit 0 |
| 0.2 | Env | DONE | `~/IsaacLab/hj` venv has all pkgs; no install |
| 0.3 | Toolchain smoke test (test.py on λ=0.99 sim) | DONE | flat λ0.99 sim test: acc .963 P .946 R .982 F1 .964 FPR .056 FNR .0175 inv_acc .962 value_mse .130 (45161 samples) |
| 1.x | Train flat: dpe, supervised, λ0.95, λ0.50, λ0.00 | DONE | via `safety_value/scripts/2_safety_analysis/train_all_baselines.sh g1_29dof_flat_unitree_ppo_6000`; ~25GB peak RAM OK, no OOM; ~2-3min/method (dpe) to ~7min (λ). Sim-test acc: λ0.99 .963 / λ0.95 .964 / λ0.50 .961 / λ0.00 .894 / dpe .843 / supervised(oracle) .967. value_mse huge for dpe (unbounded). |
| 1.y | Train avoid: dpe, supervised, λ0.95, λ0.50, λ0.00 | DONE | same script, root g1_collision_avoid_fwd_back_4999; no OOM. Sim-test acc: λ0.99 .967 / λ0.95 .962 / λ0.50 .938 / λ0.00 .911 / dpe .756 / supervised .964. FPR rises 0.007→0.119 as λ→0; dpe FPR 0.46. (λ0.99 tested separately — pre-trained, not in the script's 5.) |
| 2.1 | Write real→raw converter (+event synth, avoid contact recon) | TODO | |
| 2.2 | Build real_push / real_avoid raw + test pkl | TODO | test_proportion=1.0 |
| 3.A | Route A: test.py on real | SKIPPED | user wants inference metrics only |
| 3.B | Route B: safety_value filler + evaluate_inference on real | DONE | `fill_real_inference.py` builds per-method inference/seed_0/dataset.hdf5; `evaluate_inference.py --sa_roots real_push real_avoid --seed 0`. CSVs in results/_inference_comparison/seed_0/ |
| 4 | Report tables (real) | DONE | see §10 RESULTS |

## 10. RESULTS — real inference metrics (seed 0)

**Current numbers (2026-05-26, after data refresh + t_event fix + contact plateau; supersede the old
block below).** Avoid signal = `max(tilt@45°, graded (0.5−d)/0.5 in [−1,1] with a +1 contact plateau
when d≤0.30m)` — graded proximity warning plus a sim-like +1 on "contact" (user choice 2026-05-26).
Push = tilt@45°. Segments + event onsets = manual annotation (`logs/real_exp_replay/annotations.json`).
Eval datasets: each method's SIM model run on real obs. Metric clock starts at the annotated
**t_event** (disturbance onset); metrics computed over `[t_event, end]`. SVE uses `--value_clip 1.0`
(predicted value clipped to the bounded target range [-1,1] before MSE, so dpe's unbounded OOD
blow-ups — up to ~1.2e5 on real push — don't dominate; sign-based recall/FPR are unaffected).

Full refresh: `bash safety_value/scripts/refresh_real_eval.sh` (build_from_annotations --write → fill_real_inference
→ evaluate_inference --segment_mode event_tstar → plot_segment_diagnostics, both tasks, 6 methods).

**real_avoid (38 seg, 20 unsafe)** — clean monotonic ranking; recall and FPR both sensible now.
(The +1 plateau leaves the unsafe *set* unchanged, so recall/FPR/FNR are identical to the graded-only
signal; only SVE shifts — it rises a little because future-max now propagates +1 across the whole
pre-contact window, above the value's gradual ramp.)
| method | TempRecall | FPR | FNR | inv_acc | SVE(clip1) |
|---|---|---|---|---|---|
| supervised | 0.796 | 0.232 | 0.264 | 0.744 | 0.48 |
| λ0.99 | 0.736 | 0.305 | 0.260 | 0.730 | 0.48 |
| λ0.95 | 0.721 | 0.318 | 0.260 | 0.727 | 0.47 |
| λ0.50 | 0.488 | 0.483 | 0.231 | 0.711 | 0.58 |
| λ0.00 | 0.413 | 0.540 | 0.228 | 0.701 | 0.66 |
| dpe | 0.135 | 0.722 | 0.196 | 0.683 | 0.75 |

**real_push (17 seg, 11 unsafe)** — after push event onsets were re-annotated 2026-05-26 (moved later,
closer to the fall → shorter post-event window): recall jumped and FPR dropped, now comparable to avoid.
| method | TempRecall | FPR | SVE(clip1) |
|---|---|---|---|
| supervised | 0.844 | 0.199 | 0.27 |
| λ0.50 | 0.843 | 0.188 | 0.24 |
| λ0.95 | 0.822 | 0.203 | 0.21 |
| λ0.99 | 0.809 | 0.203 | 0.19 |
| λ0.00 | 0.743 | 0.318 | 0.26 |
| dpe | 0.621 | 0.452 | 0.44 |
(SVE with --value_clip 1.0; dpe unclipped SVE was ~3e7 from a 1.2e5 OOD value spike in
demo_push_3_seg01. Push is noisier than avoid — λ0.50/0.95/0.99 within noise on recall/FPR; λ0.50
edges them, but λ0.99 has the lowest (best-calibrated) SVE. Pre-relabel push recall ~0.39–0.60.)

**Two fixes applied 2026-05-26 (see §10b):** (1) `data_raw.hdf5` was stale — regenerated so the
avoid signal is the uncapped graded distance (not the old binary contact@0.30m) and the annotated
events are present; (2) metrics now clock from `t_event` not segment start. Together these moved
avoid λ0.99 recall 0.286→0.736 and FPR 0.803→0.305.

Scripts: `safety_value/scripts/1_data_prep/{replay_render,annotate_server,build_from_annotations}.py`,
`safety_value/scripts/2_safety_analysis/fill_real_inference.py`,
`safety_value/scripts/3_result_analysis/{evaluate_inference,plot_segment_diagnostics}.py`.

<details><summary>OLD numbers (stale data_raw + no t_event — kept for reference)</summary>

real_avoid (14 unsafe): supervised 0.321 / λ0.99 0.286 / λ0.95 0.281 / λ0.50 0.216 / λ0.00 0.184 /
dpe 0.051; FPR ~0.76–0.92. real_push: supervised 0.229, λ 0.13–0.18, dpe 0.130; FPR ~0.86.
</details>

### 10a. Per-segment diagnostic plots + why low recall / high FPR (2026-05-26)

`safety_value/scripts/3_result_analysis/plot_segment_diagnostics.py` — one figure per
(method, segment): safety_signal panel + safety_value panel (with future-max SVE
target overlaid) + a per-step TP/FP/TN/FN classification strip. Annotates
`R_temp = t_lead/t_unsafe`, per-segment value MSE, and per-segment
`FPR = FP/(FP+TN)`, plus the t_unsafe / t_lead windows. Output under each method's
`inference/seed_0/segment_diagnostics/<segment>.png`. Run:
`python safety_value/scripts/3_result_analysis/plot_segment_diagnostics.py --sa_roots real_push real_avoid --seed 0` (`--unsafe_only`, `--methods`, `--segments` to filter).

The plot greys out the excluded pre-event prefix `[0, t_event)`, marks `t_event`
(orange), and computes/shades all metrics only over `[t_event, end]` — matching
`evaluate_inference.py --segment_mode event_tstar`.

**Root cause of the original low recall / high FPR (now fixed):** the annotated
real segments are *longer than* event→event — each one includes a long pre-event
prep/standing period (avoid: ball held far away; push: standing before the shove),
up to **86%** of the segment (`demo_avoid_2_seg22`: event at 15.1s of 17.5s). The
metric labels every step by the **segment future-max** of the signal, so once the
segment eventually becomes unsafe, `future_max>0` for the *whole* segment incl. the
prep — and the value correctly stays ≤0 there, counted as FP. With the clock at
segment start this gave e.g. `push_2_front_seg00` 286 FP / 17 TN → FPR 0.94, and
`t_unsafe` ≈ full length → recall ≈ 0.07. Clocking from `t_event` excludes the prep:
the same segment's `t_unsafe` drops 302→41 and FPR/recall become meaningful.
This was the user's diagnosis: **sim segments start at the event (t_event=0); real
annotated segments don't, so t_unsafe must start at the event, not segment start.**

### 10b. Fixes applied 2026-05-26

1. **Stale `data_raw.hdf5` regenerated.** The committed `data_raw` (written 22:41/23:17)
   predated the event annotations (23:49) and used the OLD avoid signal. Confirmed by
   diffing against a fresh recompute: stored avoid signal pegged at +1.0 on binary
   contact@0.30m (and −0.0 elsewhere) vs the intended uncapped graded
   `clip((0.5−d)/0.5,−1,1)`; and every episode had **0 events**. Re-ran
   `build_from_annotations.py --task {push,avoid} --write --no_clips` (uncapped signal
   + events from `annotations.json`), then `fill_real_inference.py --task {push,avoid}`
   to rebuild every method's `inference/seed_0/dataset.hdf5`. Avoid now has 20 unsafe
   segments (was 14). **Lesson: after editing the signal/annotations, always rerun
   build_from_annotations `--write` AND fill_real_inference before evaluating** — codified in
   `safety_value/scripts/refresh_real_eval.sh` (the 4-stage pipeline; run it on any data/model/signal change).
   Final avoid signal (user choice 2026-05-26) = graded `(0.5−d)/0.5` **plus a +1 contact plateau at
   d≤0.30m** (tilt never fires on real avoid: 0/20 segs; min ball dist 0.091m, so graded-only peaked
   ~0.8 and never reached +1 — the plateau restores the sim-like +1-on-contact magnitude).

2. **`t_event` added to the metric definition** (`evaluate_inference.py`). `segment_trajectory`
   now returns `(start, end, t_event)` and takes `mode`:
   - `split` (default, **sim — unchanged**): events are segment boundaries, `t_event==start`,
     so the window is the whole segment (byte-identical metrics to before; verified the
     windows still tile `[0,T-1]`).
   - `event_tstar` (**real**): one segment per episode, `t_event` = first event index;
     all metrics (`t_unsafe`, `t_lead`, value MSE, confusion/FPR) computed over
     `[t_event, end]`; pre-event prefix excluded.
   Threaded `--segment_mode` through `compute_segment_metrics`,
   `compute_confusion_matrix_from_dataset`, `evaluate_episode/_method`, and both plot fns
   (which now mark `t_event` in orange). `plot_segment_diagnostics.py` mirrors this.
   Common format, different data generation — exactly the user's framing.

---

## 8b. End-to-end trace & obs-dim consistency (VERIFIED 2026-05-25)

`ObservationRecorder` records the **"policy"** obs group = the **actor** obs (runner obs_groups
`{"actor":["policy"], "critic":["critic"]}`, asymmetric AC). So **actor obs = recorded `policy_obs`
= safety-value input = real recorded obs**. Confirmed for both chains:

**PUSH → real_push (480-dim throughout):**
1. PPO env: `Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO-POLICY`
   (cfg `G1Flat29DofEnvCfg_UNITREE`, runner `G1Flat29DofUnitreeAsymPPORunnerCfg`,
   experiment_name `g1_29dof_flat_unitree_ppo`). Policy obs group `G1Flat29DofDeployPolicyObsCfg`:
   history_length=5, concatenate=True → **480** (base_ang_vel+proj_grav+vel_cmd+joint_pos+joint_vel+last_action, each ×5).
2. PPO policy: `logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt`
   (deployed as `unitree_flat_baseline_v4/exported/policy.onnx`, 480→29).
3. Safety rollout env: `…-PPO-SAFETY-ROLLOUT` w/ model_6000 →
   `logs/safety_analysis/g1_29dof_flat_unitree_ppo_6000/data_raw.hdf5`.
   Recorders: `policy_obs`(480), `safety_signal_balance`, `event_push`, `terminal_state`, `stability_signal_vel_track`.
4. Safety value: `results/lambda_reachability_lambda_reach_lambda_0_99`, **input_dim=480**
   (deployed as `unitree_flat_baseline_v4/exported/safety_value.onnx`, 480→1).
5. Real: obs=**480** ✓. **safety_signal recorded = tilt_only**; sim trained on
   `max(tilt, base_contact, support)`. tilt-only is a subset/proxy (a fall fires all three).

**AVOID → real_avoid (510-dim throughout):**
1. PPO env: `Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO-POLICY`
   (cfg `G1CollisionAvoidFlat29DofEnvCfg_UNITREE_FWD_BACK`, runner
   `G1CollisionAvoidFlat29DofUnitreeFwdBackAsymPPORunnerCfg`,
   experiment_name `g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo`). Policy group = flat 480
   + `_add_ball_observation_terms(observations.policy)` (ball_position 3×5 + ball_velocity 3×5 = 30) → **510**.
2. PPO policy: `logs/rsl_rl/g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/2026-05-08_17-07-38_warmstart_fwd_back_collision20/model_4999.pt`
   (deployed as `unitree_avoid_fwd_back_mocap/exported/policy.onnx`, 510→29).
3. Safety rollout env: `…-FWD-BACK-PPO-SAFETY-ROLLOUT` w/ model_4999 →
   `logs/safety_analysis/g1_collision_avoid_fwd_back_4999/data_raw.hdf5`.
   Recorders: `policy_obs`(510), `safety_signal_collision`, `event_ball_spawn`, `terminal_state`.
4. Safety value: `results/lambda_reachability_lambda_reach_lambda_0_99`, **input_dim=510**
   (deployed as `unitree_avoid_fwd_back_mocap/exported/safety_value.onnx`, 510→1).
5. Real: obs=**510** ✓. **safety_signal recorded = tilt_only (0 positives)**; sim trained on
   `max(collision[contact=+1 / dist=(0.5−d)/0.5], tilt, body_contact, support)`. Collision term
   missing from real ⇒ reconstruct contact (ball≤0.30 m → +1) and take `max(tilt, contact)` (Decision #1).

**Verdict:** obs dims are consistent across PPO-actor / safety-value-input / real-recording for both
tasks (480 push, 510 avoid). The only ground-truth gap is the **avoid collision signal** (handled by
reconstruction). Real obs reproduces deployed action & safety_value to ~1e-7 (already validated).

## 8c. thesis_ver (default/simplified G1) vs new 29-DoF — NOT comparable

`logs_thesis_ver` safety tasks use the **default G1 with single-frame obs** (group history_length=1):
flat `g1_flat_ppo_1000` input_dim **123**; avoid `g1_collision_avoid_flat_ppo_1450` input_dim **129**
(= 123 + ball_pos 3 + ball_vel 3, **single frame, no history**); rough `g1_rough_ppo_1000` 310.
New 29-DoF Unitree tasks stack **5 frames** of every term: flat **480**, avoid **510** (= 480 + ball
3×5 + 3×5 = 30). Same ball-obs helper (`_add_ball_observation_terms`, no per-term history → inherits
group history) ⇒ the 6-vs-30 delta is purely the group history (1 vs 5). ⇒ thesis sim results are NOT
an apples-to-apples baseline for the 29-DoF real eval; the 29-DoF tasks are self-contained.

## 8d. Real-data converter facts (VERIFIED 2026-05-25, via /tmp/inspect_real.py)

- **Loaders:** push demos have clean `records.npy` (structured: `numeric_columns`, `numeric_data[T,647]`,
  `string_data`=obs_key). Avoid `demo_avoid_1` is CSV-only (parse robustly, skip the `obs_key` string col).
- **Ball-obs index ordering CONFIRMED** (shift_err≈0): obs history is **frame-major, newest = LAST frame**.
  Avoid: `ball_position` block = obs[480:495], **current = obs[492:495]**; `ball_velocity` block =
  obs[495:510], **current = obs[507:510]** (base frame: x fwd, y left, z up). distance = ‖obs[492:495]‖.
- **Push events:** detect from `|root_ang_vel|` spikes (demo_push_1: baseline ~0.05, push spikes up to
  ~7; quiet ~0). Tilt(deg) from `projected_gravity` = `atan2(√(gx²+gy²),−gz)`; >45° ⇒ unsafe (matches
  recorded `safety_signal`). demo_push_1: pushes cluster 0–60 s, lone fall (tilt 53.8°) at ~600 s →
  4 unsafe steps. ⇒ contiguous segmentation at push onsets (like sim).
- **Avoid events:** ball **speed** (‖obs[507:510]‖, up to ~8.8) spikes = throw onset; ball **z**
  (obs[494], range [−1.17,1.46]) dropping to ground (~−0.7) = landing/end. Long quiet "prep" gaps
  (ball held, dist const ~2.5 m, speed~0) between throw clusters → EXCLUDE. Reconstructed contact
  (‖ball‖≤0.30 m): demo_avoid_1 only **5 steps** (ball min dist 0.28 m) — contact is sparse; the
  logged λ0.99 value fired on **901** steps (correctly warns on approaching-then-avoided balls).
  ⇒ avoid real metrics will lean on detection/value-calibration; true-positive (contact) segments are few.
  May want to also check demo_avoid_2 and revisit the 0.30 m threshold (ball can hit a limb >0.30 m from pelvis).

## 8e. Converter built + preview (2026-05-25) — open refinements

`safety_value/scripts/1_data_prep/real_to_raw.py` (preview = plots only; `--write` emits data_raw.hdf5).
Preview counts: push demo_push_1=30 events/2 unsafe(45°), push_1_front=5, push_5_left=1;
avoid demo_avoid_1=13 throws/5 contacts, **demo_avoid_2=106 throws/80 contacts (min dist 0.09 m)**.
Plots: `logs/safety_analysis/real_{push,avoid}/seg_preview/<demo>_seg.png`.

OPEN REFINEMENTS (need user):
1. **phi_max mismatch.** Deployment logged tilt `safety_signal` with **phi_max=30°** (π/6); sim training
   used **45°** (π/4). Verified: recorded max 0.79 = (53.8°−30)/30. Converter currently recomputes tilt
   at **45°** (to match the trained value). Decide 45° (match training, default) vs 30° (match deployment).
   Impact is small (push positives 2 vs 4; avoid tilt rarely dominates contact).
2. **Push "setup" trim.** Push demos have a long trivially-safe standing period before the first push
   (e.g. push_1_front: 0–46 s standing, pushes 46–54 s). Currently kept (contiguous, like sim). Option:
   trim leading/trailing quiet to first/last push ± pad, so easy negatives don't dilute FPR. Avoid
   already excludes prep.
3. **Throw-window tuning.** demo_avoid_2 (27 min) too dense to eyeball at full width — add zoomed plots
   to validate onset/landing thresholds (v_on=1.5, contact 0.30 m, etc.) before `--write`.

## 8f. PIVOT: manual segment annotation via MuJoCo replay + web tool (2026-05-25)

Rule-based segmentation was not robust enough → switched to **manual annotation**.
- **Replay:** `safety_value/scripts/1_data_prep/replay_render.py` renders each demo to mp4 via MuJoCo
  (`unitree_rl_gym` `g1_29dof.xml`, cloned at `/home/ruic/unitree_rl_gym`). Kinematic playback:
  base orientation from `root_quat` (verified **wxyz**, gravity fit err≈0), 29 joints from
  `joint_pos_0..28` — **identity mapping confirmed by eyeballing a frame** (clean standing G1, so the
  recorded joint order == g1_29dof.xml motor order). Avoid: red ball placed from `obs[492:495]`
  (base frame) rotated to world. Real-time video (25 fps, every 2nd 50 Hz sample) ⇒
  video_time_s == demo_time_s == data_index/50. Output: `logs/real_exp_replay/<demo>.mp4` + `.meta`.
  Needs `MUJOCO_GL=egl`; deps `mujoco robot_descriptions imageio imageio-ffmpeg opencv-python-headless`.
- **Annotator:** `safety_value/scripts/1_data_prep/annotate_server.py` (stdlib HTTP, Range support for scrubbing).
  http://localhost:8000 — scrub video, drag start/end markers (or `[`/`]` from playhead), Enter=add
  segment, Save; walks all 12 push + 2 avoid demos. Saves `logs/real_exp_replay/annotations.json`
  = `{demo: {task, segments: [[start_s,end_s],...]}}`. Segment seconds → data indices via ×50.
- **Status:** server running (PID 6326). 13/14 videos rendered; demo_avoid_2 (27 min) rendering.

NEXT (after user finishes annotating): (a) extend `real_to_raw.py` to consume `annotations.json`
(slice each [start,end] → one HDF5 episode; safety_signal: push=tilt@45°, avoid=max(tilt@45°,
contact@0.30m)); (b) render per-segment vids + plots for user review; (c) on approval write
`real_push`/`real_avoid` `data_raw.hdf5` + `data_processed_test.pkl`. Rule-based path in
`real_to_raw.py` (push ang_vel events, avoid throw detection) is now superseded by annotations but kept.

## 9. Key paths (post-copy)
- Sim flat task:  `logs/safety_analysis/g1_29dof_flat_unitree_ppo_6000/`
- Sim avoid task: `logs/safety_analysis/g1_collision_avoid_fwd_back_4999/`
- Existing model: `results/lambda_reachability_lambda_reach_lambda_0_99/models/last_seed_0.pt`
- Real data:      `logs/real_exp/{demo_push_*,push_*_front,push_*_left,demo_avoid_*}/<ts>_velocity/`
- Policies:       `logs/rsl_rl/g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26/model_6000.pt` (flat)
- Offline ref:    `scripts/real_exp_offline_inference.py`; `docs/real_exp_offline_inference.md`;
                  `docs/mocap_avoid_pipeline.md`
