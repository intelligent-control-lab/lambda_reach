# Supervised Regression Safety Analysis

This module implements a supervised regression approach to learning safety value functions.

## Overview

Unlike the weakly supervised approach that uses binary invariant labels with margin-based loss, this approach directly fits the predicted safety value to the ground truth sample safety value using mean squared error (MSE).

## Method

The algorithm trains a neural network to predict safety values by minimizing:

```
Loss = MSE(v(x), sample_safety_value)
     = mean((v(x) - sample_safety_value)^2)
```

Where:
- `v(x)` is the predicted safety value
- `sample_safety_value` is the ground truth safety value from the data

## Advantages

1. **Simple and direct**: Straightforward regression with no hyperparameters like margin
2. **Continuous values**: Learns the full range of safety values, not just the sign
3. **Standard loss**: Uses well-understood MSE loss

## Usage

Train with the supervised algorithm:

```bash
python safety_value/scripts/2_safety_analysis/train.py \
    --safety_analysis_root exp_name \
    --algo supervised \
    --epochs 20 \
    --batch 256
```

## Evaluation

The trainer tracks two metrics:
- **MSE**: Mean squared error between predicted and ground truth safety values
- **Accuracy**: Sign accuracy (what percentage of predictions have the correct sign relative to invariant labels)

Both metrics are computed on train and test sets during training and saved to plots and CSV files.
