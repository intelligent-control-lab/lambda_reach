# Appendix: Training, Implementation, and Hyperparameter Details

This appendix documents, for full reproducibility, (i) the RL training of the locomotion / collision‑avoidance
policies $\pi$, (ii) the collection of policy rollouts used to train the learned safety value $V$, (iii) the
training of $V$, and (iv) the protocol used to evaluate $V$ in simulation and on hardware. All numerical values
were read directly from the saved run configs (`params/env.yaml`, `params/agent.yaml`), the saved
`training_config.json` of each safety‑value run, and the source code; file paths are given so every number can be
re‑checked.

---

## A. Two robot configurations

The project uses **two distinct G1 morphologies**, and results from them are **not** directly comparable.

| Lineage | Used for | Robot model | Action dim | Obs history | Log root |
|---|---|---|---|---|---|
| **Sim‑testing (thesis)** | simulation results | default IsaacLab G1 (`g1_minimal.usd`, `G1_MINIMAL_CFG`) | **37** | single frame (history = 1) | `logs_thesis_ver/` |
| **Hardware** | real‑robot deployment | 29‑DoF Unitree G1 (`g1.usd`, `G1_29DOF_CFG`) | **29** | 5‑frame stack (history = 5) | `logs/` |

The 37‑DoF action space and the 29‑DoF action space were verified directly from the trained checkpoints
(actor output layer = 37 and 29 respectively). The 37‑DoF model actuates 12 leg + 1 torso + 10 arm + 14 finger
joints; the 29‑DoF Unitree model actuates 12 leg + 3 waist + 14 arm joints (3‑DoF wrists, no fingers).

> **Caveat (state this in the paper).** The thesis‑version simulation safety tasks use single‑frame observations
> (flat obs dim 123, rough 310, collision 129), whereas the hardware lineage stacks 5 frames (flat 480, avoid 510).
> The two are therefore **not an apples‑to‑apples baseline**; each lineage is self‑contained.

> **Hardware terrain note.** Only **flat (push)** and **collision‑avoidance** policies were deployed to hardware.
> There is **no 29‑DoF rough‑terrain hardware policy** — the only rough policy is the 37‑DoF simulation one. Any
> reference to "G1 rough on hardware" should be removed/clarified.

Common simulator settings (all tasks): NVIDIA Isaac Sim / IsaacLab; PhysX TGS solver (`solver_type=1`), gravity
$(0,0,-9.81)$, robot articulation position/velocity solver iterations 8/4, `use_fabric=true`, single GPU
(`cuda:0`). Simulation timestep **0.005 s (200 Hz)**, control **decimation 4** ⇒ **policy control rate 50 Hz
(step_dt 0.02 s)** for every task. Episode length **20 s** for every task.

---

## B. Policy training ($\pi$)

### B.1 Simulation‑testing policies (default 37‑DoF G1, `logs_thesis_ver/`)

Three tasks. Runs used: flat `g1_flat_ppo/2025-11-30_19-04-46`, rough `g1_rough_ppo/2025-12-19_02-01-37`,
collision `g1_collision_avoid_flat_ppo/2026-01-18_17-40-01`.

**Environment (shared).** Action = `JointPositionAction` over all joints (`joint_names=[".*"]`),
**scale 0.5, offset 0, `use_default_offset=true`**, no action clipping. 4096 parallel envs.
Implicit PD actuators: legs/torso stiffness 150–200, damping 5, effort 300; ankles stiffness 20, damping 2,
effort 20; arms stiffness 40, damping 10, effort 300; armature 0.01 (0.001 fingers).

**Observations** (policy group, single frame, concatenated, corruption on; uniform additive noise):

| Term | dim | noise ± | flat | rough | collision |
|---|---|---|---|---|---|
| base_lin_vel | 3 | 0.1 | ✓ | ✓ | ✓ |
| base_ang_vel | 3 | 0.2 | ✓ | ✓ | ✓ |
| projected_gravity | 3 | 0.05 | ✓ | ✓ | ✓ |
| velocity_commands | 3 | – | ✓ | ✓ | ✓ |
| joint_pos (rel) | 37 | 0.01 | ✓ | ✓ | ✓ |
| joint_vel (rel) | 37 | 1.5 | ✓ | ✓ | ✓ |
| last_action | 37 | – | ✓ | ✓ | ✓ |
| height_scan | 187 | 0.1, clip [−1,1] | – | ✓ | – |
| ball_position (robot frame) | 3 | – | – | – | ✓ |
| ball_velocity (robot frame) | 3 | – | – | – | ✓ |
| **Total obs dim** | | | **123** | **310** | **129** |

Critic observations are identical to the policy (symmetric actor‑critic; no privileged obs) for all three.

**Terrain.** flat / collision: `plane`. rough: procedural `TerrainGenerator`, curriculum on, 10 rows × 20 cols, tile
8×8 m, sub‑terrains: pyramid stairs (up/inv) 0.2/0.2 (step 0.05–0.23 m), random‑grid boxes 0.2 (height 0.05–0.2 m),
random rough 0.2 (noise 0.02–0.1 m), pyramid slope (up/inv) 0.1/0.1 (slope 0–0.4); height scanner 1.6×1.0 m grid
@0.1 m (17×11 = 187 rays) on `torso_link`. Rough uses a velocity‑based terrain‑level curriculum (`max_init_terrain_level=5`).

**Domain randomization / events** (notably light): startup `physics_material` friction set to static 0.8 / dynamic
0.6 (degenerate single‑point ranges → effectively a fixed friction, **not** a sampled spread); restitution 0.
`reset_base` pose x,y ∈ [−0.5, 0.5] m, yaw ∈ [−π, π]; `reset_robot_joints` position scale [1.0, 1.0], velocity 0
(flat) / [−1,1] differs per file. **No base‑mass randomization, no external force/torque, no push events** in any
of the three sim‑testing tasks (`add_base_mass=null`, `push_robot=null`, force/torque ranges 0).

**Commands** (`UniformVelocityCommand`, heading‑controlled, resampling 10 s): flat lin_x [0,1], lin_y [−0.5,0.5],
ang_z [−1,1]; rough lin_x [0,1], lin_y 0, ang_z [−1,1]; collision same as flat.

**Collision‑specific.** Obstacle ball: sphere radius 0.1 m, mass 0.05 kg, contact sensor (history 4). Ball is thrown
toward the robot every **3–5 s** (`spawn_ball`, interval) from distance **2–5 m**, height **0.1–1.2 m**, speed
**3–6 m/s**; `reset_ball` re‑samples the same ranges. Avoidance is curriculum‑gated: **`enable_after_iters=500`**
(locomotion warm‑up first). Extra rewards: `ball_proximity` −5.0 (std 2.0), `ball_collision` −1.0 (thr 0.1).
Torso‑contact termination is **suppressed for ball hits above 0.4 m** (`illegal_contact_unless_ball_enabled`).

**Rewards** (weights): track_lin_vel_xy 1.0, track_ang_vel_z 1.0 (rough 2.0), lin_vel_z_l2 −0.2 (rough 0),
ang_vel_xy_l2 −0.05, dof_torques_l2 −2e‑6 (rough −1.5e‑7), dof_acc_l2 −1e‑7 (rough −1.25e‑7), action_rate_l2 −0.005,
feet_air_time 0.75 (rough 0.25), flat_orientation_l2 −1.0, dof_pos_limits −1.0, termination_penalty −200,
feet_slide −0.1, joint_deviation hip/arms/fingers/torso −0.1/−0.1/−0.05/−0.1 (+ collision ball terms above).
**Terminations:** time‑out (20 s) and illegal torso contact (force > 1.0).

**PPO / network** (`rsl_rl`, `OnPolicyRunner`, `ActorCritic`):

| | flat | rough | collision |
|---|---|---|---|
| actor/critic hidden dims | [256,128,128] | [512,256,128] | [256,128,128] |
| activation | elu | elu | elu |
| init noise std | 1.0 (scalar) | 1.0 | 1.0 |
| obs normalization | off | off | off |
| value_loss_coef | 1.0 | 1.0 | 1.0 |
| clip_param | 0.2 | 0.2 | 0.2 |
| entropy_coef | 0.008 | 0.008 | 0.008 |
| learning_epochs | 5 | 5 | 5 |
| mini_batches | 4 | 4 | 4 |
| learning_rate | 1e‑3 | 1e‑3 | 1e‑3 |
| schedule | adaptive | adaptive | adaptive |
| desired_kl | 0.01 | 0.01 | 0.01 |
| gamma | 0.99 | 0.99 | 0.99 |
| lam (GAE) | 0.95 | 0.95 | 0.95 |
| max_grad_norm | 1.0 | 1.0 | 1.0 |
| steps_per_env | 24 | 24 | 24 |
| num_envs | 4096 | 4096 | 4096 |
| max_iterations | 1500 | 3000 | 1500 |
| seed | 42 | 42 | 42 |
| **total env steps** | 147.5M | 294.9M | 147.5M |
| final checkpoint | model_1499 | model_2999 | model_1499 |
| **checkpoint used for safety rollout** | **model_1000** | **model_1000** | **model_1450** |

### B.2 Hardware policies (29‑DoF Unitree G1, `logs/`)

Two deployed policies. **Flat / push:** `g1_29dof_flat_unitree_ppo/2026-04-24_13-47-26`, deployed
**`model_6000.pt`**. **Collision avoid:** `g1_collision_avoid_flat_29dof_unitree_fwd_back_ppo/2026-05-08_17-07-38_warmstart_fwd_back_collision20`,
deployed **`model_4999.pt`**.

**Environment.** Action = `JointPositionAction` over the explicit **29 Unitree joints**, **scale 0.25**, offset 0,
`use_default_offset=true`, no clip. Self‑collisions enabled.

**Observations** (policy group, **history_length 5**, concatenated, corruption on; per‑frame terms below ×5):

| Term | dim/frame | scale | noise ± | flat | avoid |
|---|---|---|---|---|---|
| base_ang_vel | 3 | 0.2 | 0.2 | ✓ | ✓ |
| projected_gravity | 3 | – | 0.05 | ✓ | ✓ |
| velocity_commands | 3 | – | – | ✓ | ✓ |
| joint_pos (rel, 29) | 29 | – | 0.01 | ✓ | ✓ |
| joint_vel (rel) | 29 | 0.05 | 1.5 | ✓ | ✓ |
| last_action | 29 | – | – | ✓ | ✓ |
| ball_position (robot frame) | 3 | – | – | – | ✓ |
| ball_velocity (robot frame) | 3 | – | – | – | ✓ |
| **per‑frame / total (×5)** | | | | **96 / 480** | **102 / 510** |

**Asymmetric actor‑critic**: the critic adds `base_lin_vel` (3) to the actor terms ⇒ critic per‑frame 99 (flat) /
105 (avoid) ⇒ **critic obs 495 (flat) / 525 (avoid)**. Deployed obs layout (real logs): base_ang_vel[0:15],
proj_grav[15:30], cmd[30:45], joint_pos[45:190], joint_vel[190:335], last_action[335:480]; avoid appends
ball_position[480:495], ball_velocity[495:510].

**Domain randomization / events** (richer than the sim‑testing lineage): startup friction static & dynamic ∈
**[0.3, 1.0]** (64 buckets); **base mass +∈[−1, 3] kg** on torso; `reset_base` x,y ∈ [−0.5,0.5], yaw ∈ [−π,π];
`reset_robot_joints` pos scale [1,1], vel [−1,1]; **`push_robot` every 5.0 s**, velocity x,y ∈ [−0.5, 0.5] m/s;
hand joints pinned to default. No external force/torque, no actuator‑gain randomization.

**Commands** (resampling 10 s). Flat: curriculum (`lin_vel_cmd_levels`) from ±0.1 to **limit lin_x ∈ [−0.5,1.0],
lin_y ∈ [−0.3,0.3], ang_z ∈ [−0.2,0.2]**. Avoid is **forward/back only**: lin_x ∈ [−0.5,1.0], **lin_y = 0,
ang_z = 0**.

**Avoid specifics.** Same ball mechanism as the sim‑testing collision task (ball radius 0.1 m, mass 0.05 kg, thrown
every 3–5 s from 2–5 m at 3–6 m/s; ball active from iter 0). Rewards added: `ball_collision` **−20.0** (the
"collision20" in the run name), `ball_proximity` −5.0 (std 2.0), `ball_approaching` −2.0 (danger_dist 2.0),
`termination_penalty` −200. The avoid policy is a **warm start**: `resume=true`, loaded from the deployed flat
baseline (`load_run=warmstart_from_unitree_flat_baseline_v4`, `model_0.pt`).

**Rewards (flat, weights):** track_lin_vel_xy 1.0 (std 0.5), track_ang_vel_z 0.5, alive 0.15, flat_orientation_l2
−5.0, dof_pos_limits −5.0, undesired_contacts −1.0, feet_slide −0.2, joint_deviation arms/waists/legs −0.1/−1.0/−1.0,
lin_vel_z_l2 −2.0, ang_vel_xy_l2 −0.05, joint_vel −1e‑3, joint_acc −2.5e‑7, action_rate −0.05, energy −2e‑5,
base_height_l2 −10.0 (target 0.78), feet_gait 0.5 (period 0.8), feet_clearance 1.0 (target 0.1 m). **Terminations:**
time‑out, base_height < 0.2 m, bad_orientation > 0.8 rad.

**PPO / network.** Both policies: `MLPModel` actor & critic hidden dims **[512, 256, 128]**, **elu**, Gaussian
policy init std 1.0, obs‑normalization off but **empirical_normalization on**. PPO: value_loss_coef 1.0,
clip_param 0.2, **entropy_coef 0.01**, learning_epochs 5, mini_batches 4, lr 1e‑3 (adaptive), gamma 0.99, lam 0.95,
desired_kl 0.01, max_grad_norm 1.0, **Adam**. steps_per_env 24, seed 42.

| | flat | avoid |
|---|---|---|
| num_envs | 8192 | 4096 |
| max_iterations (cap) | 30000 | 5000 |
| **deployed checkpoint** | model_6000 | model_4999 |
| obs / critic obs | 480 / 495 | 510 / 525 |
| warm start | from scratch | from `unitree_flat_baseline_v4` |
| total env steps @ deploy | ≈ 1.18e9 | ≈ 4.91e8 |

### B.3 Sim‑to‑sim verification and deployment (hardware only)

Each `rsl_rl` run exports `exported/policy.onnx` (+ `policy.pt`): a 3‑layer Gemm+ELU MLP, input `obs` → output
`actions`. Flat **480 → 29**; avoid **510 → 29**. Deployment dirs (external `unitree_rl_lab` repo):
`unitree_flat_baseline_v4/` (flat) and `unitree_avoid_fwd_back_mocap/` (avoid, also ships
`exported/safety_value.onnx`, 510 → 1). The avoid input grows 480 → 510 while the actor output stays 29.

The MuJoCo sim‑to‑sim stack lives in the separate `unitree_rl_lab` repo (not in this repo). What is reproducible
**in‑repo** is an **offline real‑data validation**: `scripts/real_exp_offline_inference.py` replays the deployed
observations (`logs/real_exp/.../records.csv`, obs_0..479/509) through the exported checkpoints and matches the
logged on‑robot actions and safety value to **~1e‑7**, confirming the obs/model wiring. Real‑robot metadata
confirms `robot: g1_29dof`, `control_dt: 0.02`, `obs_dim: 480`, `action_dim: 29`. For avoidance, ball position /
velocity in the base frame are produced live by an OptiTrack→NatNet→ROS2 `mocap_avoid_bridge` node at 50 Hz
(matching the IsaacLab `ball_position_in_robot_frame` / `ball_velocity_in_robot_frame` observation functions).

---

## C. Safety‑value learning

The safety value $V(x)$ predicts whether the state is in the safe (control‑)invariant set: **$V(x)\le 0$ ⇒ safe**,
**$V(x)>0$ ⇒ will become unsafe**. Framework: `safety_value/`. Pipeline:
rollout (`1_data_prep/rollout_safety_data.py`) → process (`1_data_prep/data_processing.py`) → train
(`2_safety_analysis/train.py`) → in‑sim inference (`2_safety_analysis/inference.py`) → metrics
(`3_result_analysis/evaluate_inference.py`).

### C.1 Rollout collection (training data for $V$)

`rollout_safety_data.py` registers `"<task>-SAFETY-ROLLOUT"`, loads the PPO checkpoint, and runs the
**deterministic inference policy** (`runner.get_inference_policy`) open‑loop for `--video_length` steps with
`--num_envs` parallel environments (per‑env trajectory = one episode `demo_*`); episodes reset internally on
termination. Defaults: **num_envs 2048, video_length 1000 steps, seed 42**, control 50 Hz. Output:
`logs/safety_analysis/<root>/data_raw.hdf5`. The checkpoints used per task are encoded in the root name
(flat `..._1000`, rough `..._1000`, collision `..._1450`; 29‑DoF `..._6000`, avoid `..._4999`) — i.e. the
**hardware safety values are trained on rollouts of exactly the deployed policy checkpoints**.

Recorded per episode: `policy_obs [T,obs_dim]`, the ground‑truth signal `safety_signal_{balance|collision} [T]`,
the disturbance flag `event_{push|ball_spawn} [T]`, `terminal_state [T]` (loaded but unused), and (flat only)
`stability_signal_vel_track [T]`.

**Ground‑truth safety signal $l(x)$ (>0 = unsafe), as instantiated for the 29‑DoF / hardware tasks:**

*Flat / balance* (`safety_signal_balance` = max of three terms, `hj_flat_push/.../recorders.py`):
- **Tilt:** $l_{\text{tilt}} = (\theta - \phi_{\max})/\phi_{\max}$ with $\theta = \operatorname{atan2}(\sqrt{g_x^2+g_y^2}, -g_z)$
  from projected gravity, and **$\phi_{\max} = \pi/6$ (30°)** (`flat_env_cfg.py:625`).
- **Torso contact:** $2\cdot\mathbb{1}[\,\text{force}>1.0\,]-1 \in \{-1,+1\}$ on `torso_link`.
- **Support clearance:** $(\,0.35 - \text{clearance}\,)/0.35$, clamped to $[-1,3]$ (`stance_clearance_min=0.35`).

*Avoid / collision* (`safety_signal_collision` = max of four terms, `hj_collision_avoid/.../recorders.py`):
- **Collision/distance:** $d=\lVert \text{ball}-\text{robot}\rVert$; off‑contact
  $l = \operatorname{clip}((\,d_{\text{danger}}-d\,)/d_{\text{danger}},\,-1,\,-10^{-6})$ with
  **$d_{\text{danger}} = 0.3$** (`flat_env_cfg.py:237`); on ball contact (force > **0.1**, ball height > 0.15 m) or
  within a **0.25 s latch**, $l = +1$.
- **Tilt** with **$\phi_{\max}=\pi/4$ (45°)**, **torso contact**, **support clearance** ($d_{\min}=0.25$).

**Event flags.** `event_push = 1` on the step the push fires; in the safety‑rollout env the push is an interval
event **every 4.0 s** with a Gaussian velocity kick (per‑axis $x,y \sim \mathcal N(0, 1.0)$ m/s, truncated to
$\pm 3$, **added** to current root velocity). `event_ball_spawn = 1` on the step a ball is thrown (interval 3–5 s).

**Processing** (`data_processing.py`): each episode is split into segments at the event indices
(`np.split` at `event>0`). Per segment, with $l$ the signal:
- $x=\text{obs}_{:-1}$, $x'=\text{obs}_{1:}$, `safety_signal` $= l_{:-1}$;
- **future‑max target** `sample_safety_value`$[t]=\max(l_{t:\text{end}})$ (unbounded running max);
- **invariant flag** $=\mathbb{1}[\,\text{sample\_safety\_value}\le 0\,]$;
- $x_{\text{future}}[t]=\text{obs}_{t+1:\text{end}}$, `safety_future`$[t]=l_{t+1:\text{end}}$ (full remaining trajectory; the λ‑reachability horizon source).

Stored as a 7‑tuple `(x, x_next, safety_signal, sample_safety_value, invariant_flag, x_future, safety_future)`.
The dataset is then **class‑balanced** (downsample the majority of {invariant, non‑invariant} to equal counts) and
split train/test with **`test_proportion` = 0.2**. The split/balancing use NumPy's default RNG **with no fixed
seed** (so exact sample counts are run‑dependent; the training loop itself is seeded).

**Resulting datasets:**

| Task | root | input dim | test samples (balanced) | train / test pkl |
|---|---|---|---|---|
| 29‑DoF flat | `g1_29dof_flat_unitree_ppo_6000` | 480 | 45,161 | 21.4 / 5.3 GB |
| 29‑DoF avoid | `g1_collision_avoid_fwd_back_4999` | 510 | 21,857 | 23.8 / 5.9 GB |
| thesis flat | `g1_flat_ppo_1000` | 123 | – | 6.06 / 1.50 GB |
| thesis rough | `g1_rough_ppo_1000` | 310 | – | 16.6 / 4.15 GB |
| thesis collision | `g1_collision_avoid_flat_ppo_1450` | 129 | – | 7.24 / 1.81 GB |

### C.2 Safety‑value function and training

**Network** (`safety_value/safety_analysis_algos/model.py`): MLP `Linear(d,256)–ReLU–Linear(256,256)–ReLU–Linear(256,1)`,
i.e. **hidden_dims [256,256], ReLU, scalar output**; input $d$ = task obs dim. The last layer is initialized to
**bias −2.0, zero weights** (so $V$ starts strongly safe).

**λ‑reachability (proposed method)** uses a **2‑critic ensemble** (`critic1`, `critic2`, each the MLP above).
- **Reported / deployed value:** $V(x)=\tfrac12\big(\text{critic}_1(x)+\text{critic}_2(x)\big)$.
- **Bellman target** (per sample): draw a horizon $n$ from a truncated **Geometric$(1-\lambda)$** on $[1,L]$,
  $L=\min(\text{remaining future length}, \texttt{max\_horizon}=200)$; let
  $\;m = \max_{0\le k<n} l(x_k)\;$ (running max over the next $n$ signals); draw survival
  $S\sim\text{Bernoulli}(\delta^{\,n})$; bootstrap $b=\min(\text{target}_1(x_n),\text{target}_2(x_n))$ if survived
  else $b = v_{\text{term}}$; then
  $$y \;=\; \max\big(m,\; b\big).$$
  This is the discounted HJ‑reachability backup $V(x)=\max\big(\max_k l(x_k),\,V(x_n)\big)$ with stochastic
  absorption. **$\lambda$** is the horizon discount (the contraction factor): $\lambda\!\to\!1$ ⇒ full‑horizon
  reachability, $\lambda=0$ ⇒ one‑step. **$\delta$** (the "delta" parameter) is the **per‑step survival
  probability**, default **0.99**, with absorbing value **$v_{\text{term}}=-10^6$** (both hard‑coded, not exposed
  via CLI). The bootstrap uses clipped‑double‑$Q$ pessimism ($\min$ of the two target critics), while the reported
  value uses the mean.
- **Loss** $=$ weighted sum (all weights 1.0 for the hardware lineage):
  - main regression $\;\tfrac12\mathbb E[(V_1-y)^2+(V_2-y)^2]$;
  - lower‑bound hinge $\;\mathbb E[\,\text{relu}(l_0-V_1)+\text{relu}(l_0-V_2)\,]$ (enforce $V\ge l(x)$, zero margin);
  - monotonicity hinge $\;\mathbb E[\,\text{relu}(V_1'-V_1)+\text{relu}(V_2'-V_2)\,]$ (enforce $V(x_t)\ge V(x_{t+1})$);
  - sign BCE $\;\text{BCEWithLogits}(\alpha V,\;\mathbb{1}[y>0])$, $\alpha=\texttt{alpha\_bce}=5.0$.
- **Target network:** Polyak soft update $\theta_{\text{tgt}}\leftarrow(1-\tau)\theta_{\text{tgt}}+\tau\theta$ with
  **$\tau=\texttt{target\_tau}=0.005$** every **`target_update_period`=1** step (hardware lineage).

**Baselines / ablations.**
- **dpe** (discounted policy evaluation): single critic, TD target
  $y=(1-\lambda)\,l + \lambda\max(l, V^-(x'))$ ($V^-$ = target net). $\lambda$ is **annealed** 0.9 → 0.99
  (step 0.1 of the remaining gap, triggered when the loss converges within tol 0.01 over a 500‑step window);
  target net $\tau=0.05$, update period 10.
- **supervised (oracle):** single critic, MSE to the future‑max target,
  $\mathbb E[(V(x)-\text{sample\_safety\_value})^2]$.
- **weakly_supervised:** hinge classification on the binary invariant flag only, with margin **0.1**
  (push $V\le-0.1$ for safe, $V\ge+0.1$ for unsafe). *(Trained for the thesis lineage only.)*

**Common training hyperparameters** (saved `training_config.json`): optimizer **Adam**, **lr 1e‑3**,
**batch 256**, **hidden_dims [256,256]**, **seed 0**, device cuda, **no LR schedule, no weight decay**.

**Hardware vs. thesis lineage differ** (state both):

| param | hardware (29‑DoF) | thesis (37‑DoF sim) |
|---|---|---|
| total_steps | **10000** | **2000** |
| eval_steps | 500 | 500 |
| λ‑reach loss weights (lb / mono / bce) | **1.0 / 1.0 / 1.0** | **0 / 0 / 0** (regression + absorption only) |
| λ‑reach target_tau | **0.005** | **0.05** |
| λ‑reach target_update_period | **1** | **10** |
| lr / batch / hidden / seed | 1e‑3 / 256 / [256,256] / 0 | 1e‑3 / 256 / [256,256] / 0 |

**Methods actually trained.** Hardware (each task, 6): λ@{0.99, 0.95, 0.50, 0.00}, dpe, supervised. Thesis
(each task, 8): adds λ@0.90 and weakly_supervised. Result‑dir naming: λ runs →
`lambda_reachability_lambda_reach_lambda_0_XX` (thesis tag `lambda_reach`); `dpe`, `supervised` use default tags.

### C.3 Evaluation‑data collection (testing $V$)

The metric code (`evaluate_inference.py`) is **fully offline**: per episode it needs only `safety_signal_*` (ground
truth), `safety_value` (the model's prediction along the trajectory), and `event_*` (segmentation). The *same*
metric code is therefore used in sim and on hardware; **only the data generation differs.**

**(i) Simulation — full information, same distribution as training.** `inference.py` runs the
`"<task>-SAFETY-INFERENCE"` env, which uses the **identical recorder set as the rollout env** plus one extra
recorder (`SafetyValueInference`) that evaluates the trained model on the live policy obs each step and records
`safety_value [T]` (λ‑reachability records the 2‑critic mean). Output:
`logs/safety_analysis/<root>/results/<method>/inference/seed_<s>/dataset.hdf5`, num_envs 2048, seed‑selectable.
Because it is the same simulator/task/policy as the training rollouts, **both the ground‑truth signal and the
model prediction are available from one rollout, drawn from the training distribution.**

**(ii) Hardware — partial signal + offline synthesis.** Real logs (`logs/real_exp/.../records.csv`, one row per
0.02 s) contain the network obs, the on‑robot logged `safety_value`, and a **partial** ground‑truth
`safety_signal` (`tilt_only` — only the tilt term of the sim's multi‑term max), with **no event/disturbance
column**. The full evaluation ground truth is reconstructed offline (`safety_value/scripts/1_data_prep/build_from_annotations.py`):
- **Push:** ground truth = tilt only, **recomputed from projected gravity at $\phi_{\max}=\pi/4$ (45°)**.
- **Avoid:** the real tilt never fires, so the collision term is **reconstructed from the ball position embedded in
  the obs** (current ball pos ≈ obs[492:495], newest frame last): graded
  $l_{\text{coll}}=\operatorname{clip}((0.5-d)/0.5,\,-1,1)$ **plus a +1 contact plateau when $d\le 0.30$ m**, then
  $l=\max(l_{\text{tilt@45°}}, l_{\text{coll}})$. (The +1 plateau leaves the unsafe *set* unchanged, so it affects
  only the value‑error metric, not sign‑based metrics.)
- **Events / segments:** rule‑based detection was abandoned in favor of **manual annotation**. Each demo is
  rendered to video by kinematic **MuJoCo replay** (`replay_render.py`, base pose from `root_quat` wxyz, 29 joints
  identity‑mapped, ball from obs), annotated in a web tool (`annotate_server.py`) that records, per segment, a
  `[start, end]` window **and a disturbance‑onset event marker** → `logs/real_exp_replay/annotations.json`.
  `build_from_annotations.py --write` slices each window into one HDF5 episode and places a single event flag at the
  annotated onset. Counts: **12 push demos → 17 segments (11 unsafe); 2 avoid demos → 38 segments (20 unsafe).**
- **Predictions:** `fill_real_inference.py` runs each method's **sim‑trained** checkpoint on the real obs to produce
  `safety_value [T]` and writes the **same `dataset.hdf5` format** the sim path produces. Cross‑check: the λ@0.99
  reconstruction matches the on‑robot logged safety value to ~1e‑7.

The full hardware refresh is one script: `safety_value/scripts/refresh_real_eval.sh`
(build_from_annotations → fill_real_inference → evaluate_inference `--segment_mode event_tstar --value_clip 1.0` →
plot_segment_diagnostics).

### C.4 Metric definitions (`evaluate_inference.py`)

Each segment yields `(start, end, t_event)`; **all metrics are computed over the window $[t_{\text{event}}, \text{end}]$.**
- **`split` mode (simulation, default):** events are segment boundaries, so $t_{\text{event}}=\text{start}$ and the
  window is the whole segment (identical to the offline segmentation).
- **`event_tstar` mode (hardware/annotated):** the whole episode is one segment and $t_{\text{event}}$ is the
  annotated onset; the pre‑event prefix is excluded. *(This was essential: annotated real segments include a long
  pre‑disturbance standing/prep period — up to 86% of a segment — which otherwise inflates FPR and $t_{\text{unsafe}}$.)*

Definitions (signal $>0$ ⇒ unsafe; prediction $>0$ ⇒ warns):
- **$t_{\text{unsafe}}$** = steps from $t_{\text{event}}$ to the first $\text{safety\_signal}>0$ (undefined if the
  segment never becomes unsafe).
- **$t_{\text{lead}}$** = (first unsafe step) − (first step where $\text{safety\_value}>0$ at/after $t_{\text{event}}$
  and before the unsafe step); $0$ if it never warns in time.
- **temporal recall** $= t_{\text{lead}} / t_{\text{unsafe}}$, aggregated over unsafe segments (mean/std/min/max/median).
- **detection rate** = fraction of unsafe segments with $t_{\text{lead}}>0$.
- **sample value error (SVE)** = per‑segment $\text{MSE}\big(\text{safety\_value},\ \max(\text{future signal})\big)$,
  aggregated over all segments. With **`--value_clip 1.0`** the prediction is clipped to $[-1,1]$ before the MSE (the
  future‑max target is bounded to the signal range), so the unbounded out‑of‑distribution blow‑ups of `dpe` (up to
  ~1.2e5 on real push) do not dominate; clipping preserves sign, so recall/FPR are unaffected.
- **invariant confusion matrix** (per step, using the window's segment‑aware future‑max): positive class = **safe /
  invariant**, with $\text{pred}=\mathbb1[V\le0]$, $\text{label}=\mathbb1[\text{future\_max}\le0]$, giving
  accuracy, precision, recall, F1, **FPR** $=FP/(FP+TN)$, **FNR**.
  - **Convention note (state in paper):** because the positive class is "safe", the reported `invariant_fpr` is the
    rate at which a *will‑become‑unsafe* step is predicted safe (i.e. a miss), not a false‑alarm rate. State the
    polarity explicitly when reporting FPR.

For completeness, `test.py` reports, on the held‑out processed test pkl: per‑sample classification
($\text{pred}=\mathbb1[V\le0]$ vs invariant_flag) → accuracy/precision/recall/F1/FPR/FNR, and
`value_mse` $=\mathbb E[(V-\text{sample\_safety\_value})^2]$.

Outputs: `results/_inference_comparison/seed_<s>/inference_evaluation_seed_<s>.csv` (all metrics per method) plus
per‑(method, segment) diagnostic figures (`plot_segment_diagnostics.py`).

---

## D. Compute and software

Single **NVIDIA RTX 4090 (24 GB)**, driver 580.82.09. **PyTorch 2.7.0+cu128**, CUDA, Python 3.11
(IsaacLab 0.46, `rsl_rl`, `safety-value` 0.1.0). Safety‑value training time (29‑DoF, 10k steps): dpe ≈ 2–3 min, supervised
similar, λ‑reachability ≈ 7 min/run (the per‑sample variable‑horizon target loop dominates); peak RAM ≈ 25 GB (the
21–24 GB train pkls are loaded fully into RAM, so methods are trained sequentially). Policy training uses 4096–8192
parallel Isaac Sim envs on the same GPU.

---

## E. Known inconsistencies / things to reconcile before submission

1. **Sim‑vs‑real flat tilt threshold.** Sim *training* uses $\phi_{\max}=30°$; the on‑robot signal was also logged
   at 30°; but the offline **real‑push reconstruction uses 45°**. These should be unified (or the discrepancy
   justified) since $V$ was trained against the 30° signal.
2. **Sim‑vs‑real avoid danger distance.** Sim training uses $d_{\text{danger}}=0.3$ m for the graded collision ramp
   plus a true contact sensor; the **real reconstruction uses a 0.5 m graded ramp** plus a +1 plateau at 0.30 m. The
   graded denominators differ (0.3 vs 0.5).
3. **37‑DoF (thesis) vs 29‑DoF (hardware)** results are not directly comparable (different morphology, obs history,
   and safety‑value training budget/losses) — see §A and the §C.2 table.
4. **Unseeded train/test split** in `data_processing.py` (NumPy default RNG) — fix a seed for exact reproducibility.
5. **FPR polarity** in `evaluate_inference.py` (positive class = safe) — report the convention explicitly.
