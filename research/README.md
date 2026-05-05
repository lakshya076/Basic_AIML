# Research Experiments

This directory contains standalone scripts to conduct research experiments using the custom Neural Network framework.

## How to run

Ensure you are in the project root directory, then run any of the scripts:

```bash
python research/1_optimizers_comparison.py
python research/2_vanishing_gradient.py
python research/3_cnn_invariance.py
python research/4_initialization_study.py
```

## Experiment Details

### 1. Optimizers Comparison

- **Topic:** Comparative Analysis of Optimization Algorithms.
- **Focus:** Comparison between `SGD` (Stochastic Gradient Descent) and `Adam` optimizers.
- **Output:** `research/optimizers_comparison.png`

### 2. Vanishing Gradient

- **Topic:** The "Vanishing Gradient" Investigation.
- **Focus:** Demonstrates how `Sigmoid` activation fails in deep networks compared to `ReLU`.
- **Output:** `research/vanishing_gradient.png`

### 3. CNN Spatial Invariance

- **Topic:** CNN Architecture & Spatial Invariance Study.
- **Focus:** Evaluates the impact of `MaxPool2D` on a model's ability to handle shifted images.
- **Output:** `research/cnn_invariance.png`

### 4. Initialization Study

- **Topic:** Weight Initialization & Convergence Speed.
- **Focus:** Compares `Random`, `Xavier`, and `He` initialization methods.
- **Output:** `research/initialization_study.png`
