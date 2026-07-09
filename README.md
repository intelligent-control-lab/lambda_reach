# $\lambda$-Reachability: Geometric-Horizon Safety Bellman Equations for Humanoid Safety

This repository contains the Isaac Lab environments, safety-value learning code, and hardware deployment notes used for the humanoid experiments in:

> $\lambda$-Reachability: Geometric-Horizon Safety Bellman Equations for Humanoid Safety

The code is organized around one workflow: train a deployable humanoid policy in Isaac Lab, roll out that fixed policy to collect safety-labeled trajectories, train safety value functions from those rollouts, and export the policy and safety value model for hardware evaluation.

## Repository Layout

- `source/hj_humanoid/`: Isaac Lab external project that registers the humanoid locomotion, push, and collision-avoidance environments.
- `safety_value/`: safety rollout processing, value-function definitions, training, testing, inference, and evaluation scripts. See `safety_value/README.md`.
- `deploy/push/`: end-to-end Unitree G1 push experiment procedure, including policy training, safety-value training, ONNX export, sim-to-sim validation, and hardware deployment.
- `deploy/avoid/`: end-to-end Unitree G1 collision-avoidance procedure, including MoCap bridge setup and hardware deployment.
- `deploy/tools/`: export and offline analysis utilities for deployed safety value models.

## Installation

Install Isaac Lab following the official Isaac Lab installation guide. Activate the Isaac Lab Python environment before running this repository's scripts. For example, if Isaac Lab is installed in a virtual environment:

```bash
source /path/to/isaaclab/venv/bin/activate
```

From the repository root, install both the Isaac Lab extension and the safety-value package in editable mode:

```bash
python -m pip install -e source/hj_humanoid
python -m pip install -e safety_value
```

Verify that the local Isaac Lab tasks are registered:

```bash
python scripts/list_envs.py
```

The main task families are:

```text
Isaac-Velocity-Flat-G1-PPO
Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO
Isaac-Collision-Avoid-Flat-G1-29DOF-UNITREE-FWD-BACK-PPO
```

The scripts take the base task name above and append internal suffixes such as `-POLICY`, `-SAFETY-ROLLOUT`, and `-SAFETY-INFERENCE`.

## Workflow

### 1. Train a Policy

Train or reuse an RSL-RL policy checkpoint. For the 29-DoF Unitree flat locomotion task used by the push experiment:

```bash
python safety_value/scripts/0_policy/train.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --num_envs 8192 \
  --max_iterations 15000 \
  --seed 42 \
  --headless
```

Checkpoints are written under `logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt`.

### 2. Collect Safety Rollouts

Choose one fixed policy checkpoint for all downstream steps:

```bash
POLICY_RUN=<your_policy_run>
POLICY_STEP=<your_policy_step>
POLICY_CKPT=logs/rsl_rl/g1_29dof_flat_unitree_ppo/${POLICY_RUN}/model_${POLICY_STEP}.pt
SA_ROOT=g1_29dof_flat_unitree_ppo_${POLICY_STEP}

python safety_value/scripts/1_data_prep/rollout_safety_data.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --checkpoint ${POLICY_CKPT} \
  --safety_analysis_root ${SA_ROOT} \
  --num_envs 4096 \
  --video_length 1000 \
  --seed 42 \
  --headless
```

This writes raw rollout data to `logs/safety_analysis/<SA_ROOT>/data_raw.hdf5`.

### 3. Process Data and Train Safety Values

Process the raw rollout into train/test datasets:

```bash
python safety_value/scripts/1_data_prep/data_processing.py \
  --safety_analysis_root logs/safety_analysis/${SA_ROOT} \
  --raw_dataset_name data_raw.hdf5 \
  --test_proportion 0.2
```

Train the $\lambda$-reachability safety value function:

```bash
python safety_value/scripts/2_safety_analysis/train.py \
  --safety_analysis_root ${SA_ROOT} \
  --algo lambda_reachability \
  --model_tag lambda_reachability \
  --run_name lambda_reach_lambda_0_99 \
  --lambda_param 0.99 \
  --max_horizon 200 \
  --total_steps 10000 \
  --eval_steps 500 \
  --batch 256 \
  --lr 1e-3 \
  --hidden_dims 256 256 \
  --seed 0 \
  --device cuda
```

See `safety_value/README.md` for the full training, testing, inference, and evaluation commands, including the `dpe`, `supervised`, and `weakly_supervised` baselines.

### 4. Export for Deployment

Export the RSL-RL policy. The `play.py` wrapper loads the checkpoint, exports
the RSL-RL actor to `exported/policy.onnx` and `exported/policy.pt`, then starts
policy playback:

```bash
python safety_value/scripts/0_policy/play.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --checkpoint ${POLICY_CKPT} \
  --num_envs 1 \
  --headless
```

You can stop the script after the export message if you only need the deployment
model.

Export the trained safety value function:

```bash
SV_RUN=lambda_reachability_lambda_reach_lambda_0_99
SV_DIR=logs/safety_analysis/${SA_ROOT}/results/${SV_RUN}

python deploy/tools/export_onnx.py \
  --safety_value_dir ${SV_DIR} \
  --output ${SV_DIR}/exported/safety_value.onnx
```

The policy ONNX is written under the policy run's `exported/` directory, and the safety value ONNX is written under the safety-value run directory.

### 5. Deploy or Reproduce the Hardware Pipelines

Use the experiment-specific deployment guides:

- Push experiment: `deploy/push/README.md`
- Collision avoidance experiment: `deploy/avoid/README.md`

Those guides include the exact task names, observation dimensions, ONNX export expectations, `unitree_rl_lab` configuration, and real-robot launch steps.

## Development Notes

This repository was built as an Isaac Lab external project. If your IDE cannot resolve Isaac Lab or Omniverse modules, add the local extension to `python.analysis.extraPaths`:

```json
{
  "python.analysis.extraPaths": [
    "<path-to-this-repo>/source/hj_humanoid"
  ]
}
```

Optional formatting:

```bash
python -m pip install pre-commit
pre-commit run --all-files
```

## License

Unless a file contains a different license notice, code in this repository is
licensed under the PolyForm Noncommercial License 1.0.0. Documentation, written
research materials, figures, plots, and other non-code repository materials are
licensed under CC BY-NC 4.0 unless otherwise noted.

Commercial use is not permitted without prior written permission. For commercial
licensing, contact ruic3@andrew.cmu.edu.

This repository is source-available for noncommercial academic and research use;
it is not OSI open source because commercial use is restricted. Files derived
from third-party projects and third-party dependencies remain under their own
licenses.

## Citation

```bibtex
@misc{chen2026lambdareachability,
      title={$\lambda$-Reachability: Geometric-Horizon Safety Bellman Equations for Humanoid Safety},
      author={Rui Chen and Shangtao Li and Yifan Sun and Changliu Liu},
      year={2026},
      eprint={2606.16022},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.16022},
}
```
