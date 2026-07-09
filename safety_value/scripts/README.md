# Safety Analysis Pipeline Automation

This directory contains the master pipeline script that automates the entire safety analysis workflow.

## Overview

The `run_pipeline.py` script orchestrates training, testing, inference, and evaluation of multiple safety analysis methods across multiple tasks.

## Pipeline Steps

### 1. Training
Trains each selected algorithm on each selected task using processed safety data.

**Example command generated:**
```bash
python safety_value/scripts/2_safety_analysis/train.py \
    --safety_analysis_root g1_flat_ppo_1000 \
    --algo dpe \
    --epochs 20 \
    --batch 256
```

### 2. Testing
Tests each trained model on held-out test data and computes performance metrics (accuracy, precision, recall, F1 score).

**Example command generated:**
```bash
python safety_value/scripts/2_safety_analysis/test.py \
    --safety_analysis_root g1_flat_ppo_1000 \
    --model_subdir dpe_default_lam0.9
```

**Results:** Test results are saved to `logs/safety_analysis/{task}/{model_subdir}/test/`

**Aggregation:** After testing all methods, results are aggregated into `logs/safety_analysis/{task}/evaluation/testing/aggregated_results.json`

### 3. Inference
Runs trained models in Isaac Sim environments to collect safety value predictions during policy execution.

**Example command generated:**
```bash
python safety_value/scripts/2_safety_analysis/inference.py \
    --task Isaac-Velocity-Flat-G1-PPO \
    --safety_analysis_root g1_flat_ppo_1000 \
    --model_subdir dpe_default_lam0.9 \
    --checkpoint /home/ruic/hj_humanoid/logs/rsl_rl/g1_flat_ppo/2025-11-30_19-04-46/model_1000.pt \
    --video --video_length 1000 \
    --seed 2345
```

**Results:** Inference results (HDF5 datasets and videos) are saved to `logs/safety_analysis/{task}/{model_subdir}/inference/seed_{seed}/`

### 4. Evaluation
Evaluates and compares inference results across all methods using temporal recall metrics.

**Example command generated:**
```bash
python safety_value/scripts/3_result_analysis/evaluate_inference.py \
    --sa_roots g1_flat_ppo_1000 \
    --methods dpe_default_lam0.9 supervised_default \
    --seed 2345 \
    --plot_worst 10 --plot_best 10
```

**Results:** Evaluation results and comparison plots are saved to `logs/safety_analysis/{task}/evaluation/inference/`

## Usage

### Interactive Mode (Recommended)

Simply run the script and answer the prompts:

```bash
python safety_value/scripts/run_pipeline.py
```

You'll be asked to select:
1. Which tasks to process
2. Which algorithms to use
3. Which steps to execute

The script will then display a complete execution plan and ask for confirmation before proceeding.

### Command-Line Mode

You can also specify all options via command-line arguments:

```bash
python safety_value/scripts/run_pipeline.py \
    --tasks g1_flat_ppo_1000 \
    --algorithms dpe supervised weakly_supervised \
    --steps train test inference evaluate \
    --seed 2345
```

### Dry Run Mode

Preview all commands without executing them:

```bash
python safety_value/scripts/run_pipeline.py --dry-run
```

This is useful for:
- Checking what commands will be run
- Verifying task configurations
- Debugging the pipeline

## Task Configuration

Tasks are configured in the `TASK_CONFIGS` dictionary at the top of the script:

```python
TASK_CONFIGS = {
    "g1_flat_ppo_1000": {
        "task": "Isaac-Velocity-Flat-G1-PPO",
        "checkpoint": "/home/ruic/hj_humanoid/logs/rsl_rl/g1_flat_ppo/2025-11-30_19-04-46/model_1000.pt",
    },
    # Add more tasks here...
}
```

Each task requires:
- **key**: Safety analysis root (directory name under `logs/safety_analysis/`)
- **task**: Isaac Sim task name
- **checkpoint**: Path to trained policy checkpoint

### Adding New Tasks

1. Train and collect safety data for your task
2. Process the data using `safety_value/scripts/1_data_prep/data_processing.py`
3. Add task configuration to `TASK_CONFIGS` in `run_pipeline.py`. A task may provide either an explicit `checkpoint` path or a `runner` plus `checkpoint_step`, which lets the pipeline select the latest matching policy run.
4. Run the pipeline!

## Available Algorithms

- `dpe`: Discounted Policy Evaluation (temporal difference learning)
- `weakly_supervised`: Weakly supervised learning (margin-based classification)
- `supervised`: Supervised regression (direct value fitting)
- `lambda_reachability`: Lambda-reachability safety value learning

## Default Parameters

### Training Parameters
```python
{
    "epochs": 20,
    "batch": 256,
    "lr": 1e-3,
}
```

### Inference Parameters
```python
{
    "video": True,
    "video_length": 1000,
    "num_envs": 1,
}
```

These can be modified in the script's `DEFAULT_TRAIN_PARAMS` and `DEFAULT_INFERENCE_PARAMS` dictionaries.

## Output Structure

```
logs/safety_analysis/{task}/
├── data_processed_train.pkl          # Processed training data
├── data_processed_test.pkl           # Processed test data
├── {algo}_default_{params}/          # Algorithm-specific directory
│   ├── models/                       # Trained model checkpoints
│   │   ├── step_500.pt
│   │   ├── step_1000.pt
│   │   └── ...
│   ├── train/                        # Training metrics and plots
│   │   ├── plots/
│   │   └── training_metrics.csv
│   ├── test/                         # Testing results
│   │   ├── results_step_{N}.json
│   │   └── summary_step_{N}.txt
│   └── inference/                    # Inference results
│       └── seed_{seed}/
│           ├── dataset.hdf5
│           └── videos/
└── evaluation/                       # Aggregated evaluation results
    ├── testing/
    │   └── aggregated_results.json
    └── inference/
        └── comparison_plots/
```

## Error Handling

The pipeline includes robust error handling:
- Each step reports success/failure
- Failed steps prompt whether to continue
- Final summary shows success rate
- Dry-run mode for debugging

## Tips

1. **Start with a dry run**: Use `--dry-run` to verify everything looks correct
2. **Test incrementally**: Run one step at a time during initial setup
3. **Check configurations**: Verify task configs point to correct checkpoints
4. **Monitor progress**: The script provides detailed progress output with colors
5. **Review results**: Check aggregated results after each major step

## Example Workflows

### Full Pipeline for One Task
```bash
python safety_value/scripts/run_pipeline.py \
    --tasks g1_flat_ppo_1000 \
    --algorithms dpe supervised \
    --steps train test inference evaluate
```

### Only Training and Testing
```bash
python safety_value/scripts/run_pipeline.py \
    --tasks g1_flat_ppo_1000 g1_flat_ppo_2000 \
    --algorithms dpe supervised weakly_supervised \
    --steps train test
```

### Inference Only (Models Already Trained)
```bash
python safety_value/scripts/run_pipeline.py \
    --tasks g1_flat_ppo_1000 \
    --algorithms dpe supervised \
    --steps inference evaluate
```

## Troubleshooting

### "Task not found in TASK_CONFIGS"
Add your task configuration to the `TASK_CONFIGS` dictionary in the script.

### "No checkpoints found"
Ensure the training step completed successfully and checkpoints were saved.

### "Dataset not found"
Verify that data has been processed using `safety_value/scripts/1_data_prep/data_processing.py`.

### Isaac Sim errors during inference
Ensure Isaac Sim is properly installed and the task name is correct in `TASK_CONFIGS`.
