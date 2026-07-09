# Safety Value Learning

This package contains the safety-value workflow used by the humanoid experiments. It collects safety rollouts, processes datasets, defines several safety value trainers, runs inference, and evaluates simulation or hardware logs.

## Install

Run from the repository root after activating the Python environment that has Isaac Lab installed:

```bash
source /path/to/isaaclab/venv/bin/activate
python -m pip install -e source/hj_humanoid
python -m pip install -e safety_value
```

Verify the package import:

```bash
python - <<'PY'
from safety_value.safety_analysis_algos.model import build_mlp
from safety_value.safety_analysis_algos.dataset import HJValueDataset
from safety_value.safety_analysis_algos.dpe import DiscountedPolicyEvaluationTrainer
from safety_value.safety_analysis_algos.lambda_reachability import LambdaReachabilityTrainer
from safety_value.safety_analysis_algos.supervised import SupervisedTrainer

print("safety_value import OK")
PY
```

## Structure

```text
safety_value/
├── scripts/
│   ├── 0_policy/             # RSL-RL policy train/play/export wrappers
│   ├── 1_data_prep/          # simulated and real data conversion
│   ├── 2_safety_analysis/    # train, test, and inference entry points
│   ├── 3_result_analysis/    # aggregate metrics and diagnostic plots
│   ├── config/               # pipeline algorithm configs
│   └── run_pipeline.py       # multi-step pipeline automation
└── safety_analysis_algos/
    ├── dpe/                  # Discounted Policy Evaluation baseline
    ├── lambda_reachability/  # lambda-reachability trainer
    ├── supervised/           # supervised value regression
    ├── weakly_supervised/    # margin-based weak supervision
    ├── dataset.py
    ├── loss.py
    └── model.py
```

## Workflow

The scripts are intended to run from the repository root. Policy/data scripts accept the base task name, then append the appropriate Isaac Lab registration suffix internally.

### 1. Train or Reuse a Policy

```bash
python safety_value/scripts/0_policy/train.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --num_envs 8192 \
  --max_iterations 15000 \
  --seed 42 \
  --headless
```

Policy checkpoints are saved to `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`.

### 2. Roll Out Safety Data

```bash
POLICY_CKPT=logs/rsl_rl/g1_29dof_flat_unitree_ppo/<run>/model_<step>.pt
SA_ROOT=g1_29dof_flat_unitree_ppo_<step>

python safety_value/scripts/1_data_prep/rollout_safety_data.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --checkpoint ${POLICY_CKPT} \
  --safety_analysis_root ${SA_ROOT} \
  --num_envs 4096 \
  --video_length 1000 \
  --seed 42 \
  --headless
```

Output:

```text
logs/safety_analysis/<SA_ROOT>/data_raw.hdf5
```

### 3. Process the Dataset

```bash
python safety_value/scripts/1_data_prep/data_processing.py \
  --safety_analysis_root logs/safety_analysis/${SA_ROOT} \
  --raw_dataset_name data_raw.hdf5 \
  --test_proportion 0.2
```

Outputs:

```text
logs/safety_analysis/<SA_ROOT>/data_processed_train.pkl
logs/safety_analysis/<SA_ROOT>/data_processed_test.pkl
```

By default, event-based segmentation is enabled when event flags exist in the dataset. Use `--no_segment_by_event` to disable it.

### 4. Train Safety Value Functions

Lambda-reachability:

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

Discounted Policy Evaluation baseline:

```bash
python safety_value/scripts/2_safety_analysis/train.py \
  --safety_analysis_root ${SA_ROOT} \
  --algo dpe \
  --lam 0.9 \
  --lambda_update_steps 500 \
  --lambda_max 0.99 \
  --total_steps 10000 \
  --batch 256 \
  --device cuda
```

Supervised baseline:

```bash
python safety_value/scripts/2_safety_analysis/train.py \
  --safety_analysis_root ${SA_ROOT} \
  --algo supervised \
  --total_steps 10000 \
  --batch 256 \
  --device cuda
```

Training outputs are saved under:

```text
logs/safety_analysis/<SA_ROOT>/results/<model_subdir>/
  training_config.json
  models/
  train/
```

### 5. Test and Run Inference

Test a trained model on the held-out processed dataset:

```bash
python safety_value/scripts/2_safety_analysis/test.py \
  --safety_analysis_root ${SA_ROOT} \
  --model_subdir lambda_reachability_lambda_reach_lambda_0_99
```

Run Isaac Lab inference with a trained safety value model:

```bash
python safety_value/scripts/2_safety_analysis/inference.py \
  --task Isaac-Velocity-Flat-G1-29DOF-UNITREE-PPO \
  --safety_analysis_root ${SA_ROOT} \
  --model_subdir lambda_reachability_lambda_reach_lambda_0_99 \
  --checkpoint ${POLICY_CKPT} \
  --num_envs 2048 \
  --video \
  --video_length 1000 \
  --headless
```

### 6. Evaluate Inference Outputs

```bash
python safety_value/scripts/3_result_analysis/evaluate_inference.py \
  --sa_roots ${SA_ROOT} \
  --methods lambda_reachability_lambda_reach_lambda_0_99 supervised dpe \
  --seed 2345 \
  --base_dir logs/safety_analysis \
  --plot_worst 10 \
  --plot_best 10
```

## Pipeline Automation

`safety_value/scripts/run_pipeline.py` can generate and run train, test, inference, and evaluation commands from `safety_value/scripts/config/pipeline_algorithms.json`.

Preview commands without executing them:

```bash
python safety_value/scripts/run_pipeline.py \
  --config safety_value/scripts/config/pipeline_algorithms.json \
  --tasks ${SA_ROOT} \
  --steps train test \
  --dry-run
```

Run the full configured pipeline:

```bash
python safety_value/scripts/run_pipeline.py \
  --config safety_value/scripts/config/pipeline_algorithms.json \
  --tasks ${SA_ROOT} \
  --steps train test inference evaluate \
  --inference_mode 1 \
  --suffix exp1
```

## Export for Hardware

Use `deploy/tools/export_onnx.py` to export a trained safety value model:

```bash
SV_DIR=logs/safety_analysis/${SA_ROOT}/results/lambda_reachability_lambda_reach_lambda_0_99

python deploy/tools/export_onnx.py \
  --safety_value_dir ${SV_DIR} \
  --output ${SV_DIR}/exported/safety_value.onnx
```

See `deploy/push/README.md` and `deploy/avoid/README.md` for complete hardware deployment procedures.
