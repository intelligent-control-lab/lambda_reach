# Safety Analysis Algorithms

This package provides multiple algorithms for learning safety value functions from trajectory data.

## Algorithms

### Discounted Policy Evaluation (DPE)
Implemented in `safety_analysis_algos/dpe/`, this algorithm uses temporal difference learning with a discount factor to estimate safety values. Similar to policy evaluation in reinforcement learning, it learns to predict future safety through bootstrapping.

**Key features:**
- Lambda annealing for curriculum learning
- Optional target networks (DQN-style)
- Temporal difference loss

### Lambda-Reachability
Implemented in `safety_analysis_algos/lambda_reachability/`, this is the main safety value trainer used by the paper experiments.

**Key features:**
- Geometric-horizon Bellman target controlled by `lambda_param`
- Two-critic value model for the deployed lambda-reachability runs
- Hinge and BCE terms for safety-value sign and monotonicity structure

### Weakly Supervised Learning
Located in `weakly_supervised/`, this algorithm uses direct weakly supervised learning with margin-based loss to classify states as safe (invariant) or unsafe.

**Key features:**
- Margin-based hinge loss
- Direct binary classification using invariant labels
- Simpler training dynamics

### Supervised Regression
Located in `supervised/`, this algorithm directly fits the predicted safety value to the sample safety value using mean squared error (MSE) regression.

**Key features:**
- Simple MSE loss
- Direct value regression from ground truth safety values
- Most straightforward supervised approach

## Package Structure

```
safety_analysis_algos/
├── __init__.py                     # Package exports
├── model.py                        # MLP architecture
├── dataset.py                      # PyTorch dataset for trajectory data
├── loss.py                         # Loss function implementations
├── dpe/                            # Temporal difference algorithm
│   ├── __init__.py
│   └── trainer.py
├── lambda_reachability/            # Lambda-reachability algorithm
│   ├── __init__.py
│   └── trainer.py
├── weakly_supervised/              # Weakly supervised learning algorithm
│   ├── __init__.py
│   └── trainer.py
└── supervised/                     # Supervised regression algorithm
    ├── __init__.py
    └── trainer.py
```

## Usage

See `safety_value/scripts/2_safety_analysis/train.py` for training examples.

**Discounted Policy Evaluation:**
```python
from safety_value.safety_analysis_algos.dpe import DiscountedPolicyEvaluationTrainer
from safety_value.safety_analysis_algos.dataset import HJValueDataset

trainer = DiscountedPolicyEvaluationTrainer(
    input_dim=input_dim,
    train_dataset=train_ds,
    test_dataset=test_ds,
    initial_lambda=0.9,
    enable_lambda_annealing=True,
)
trainer.train(epochs=20, ckpt_dir='./checkpoints')
```

**Lambda-Reachability:**
```python
from safety_value.safety_analysis_algos.lambda_reachability import LambdaReachabilityTrainer
from safety_value.safety_analysis_algos.dataset import HJValueDataset

trainer = LambdaReachabilityTrainer(
    input_dim=input_dim,
    train_dataset=train_ds,
    test_dataset=test_ds,
    lambda_param=0.99,
    max_horizon=200,
)
trainer.train(epochs=20, ckpt_dir='./checkpoints')
```

**Weakly Supervised Learning:**
```python
from safety_value.safety_analysis_algos.weakly_supervised import WeaklySupervisedTrainer
from safety_value.safety_analysis_algos.dataset import HJValueDataset

trainer = WeaklySupervisedTrainer(
    input_dim=input_dim,
    train_dataset=train_ds,
    test_dataset=test_ds,
    margin=0.1,
)
trainer.train(epochs=20, ckpt_dir='./checkpoints')
```

**Supervised Regression:**
```python
from safety_value.safety_analysis_algos.supervised import SupervisedTrainer
from safety_value.safety_analysis_algos.dataset import HJValueDataset

trainer = SupervisedTrainer(
    input_dim=input_dim,
    train_dataset=train_ds,
    test_dataset=test_ds,
)
trainer.train(epochs=20, ckpt_dir='./checkpoints')
```
