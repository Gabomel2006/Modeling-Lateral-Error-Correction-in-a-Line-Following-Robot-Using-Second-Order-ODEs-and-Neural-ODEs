# Modeling Lateral Error Correction in a Line-Following Robot Using Second-Order ODEs and Neural ODEs

**Author:** Gabriel Melgarejo
**Course:** Math 2A — Differential Equations, Foothill College

## Overview

This project models how a line-following robot corrects its position under **PD (Proportional-Derivative) control**. By combining Newton's second law with PD control theory, the robot's lateral error `e(t)` is shown to satisfy a second-order linear homogeneous ODE:

```
e''(t) + kd·e'(t) + kp·e(t) = 0
```

The equation is solved analytically using the characteristic equation method, then verified two ways: numerically in MATLAB, and computationally with a **Neural ODE** trained from scratch in PyTorch.

## Key Results

- With parameters `m = 1`, `kp = 4`, `kd = 2`, the system is **underdamped** (discriminant `D = -12 < 0`), meaning the robot **oscillates across the line** before settling.
- The particular solution:
  ```
  e(t) = e^(-t) · (0.1·cos(√3·t) + (0.1/√3)·sin(√3·t))
  ```
- Predicted lateral error at `t = 2s`: **e(2) ≈ -0.0153 m** (robot has crossed the line), confirmed by MATLAB's `ode45` solver.
- A Neural ODE trained on 1000 data points (no knowledge of the underlying physics) independently learned the same dynamics, predicting `e(2) ≈ -0.0146 m` — a difference of **0.68 mm** from the analytical result.

## Repo Structure

```
├── paper.pdf              # Full research paper
├── matlab/
│   ├── particular_solution.m      # Plots e(t) for the underdamped case (Section 6.7)
│   └── underdamped_prediction.m   # Plots e(t) with prediction at t = 2s (Section 7.3)
└── python/
    └── neural_ode.py               # Trains a Neural ODE to learn e(t) from data (Section 10)
```

## How to Run

### MATLAB
Requires MATLAB with base plotting/export capabilities (no special toolboxes needed).

```matlab
% From the matlab/ directory
run('particular_solution.m')
run('underdamped_prediction.m')
```

Each script generates a `.png` plot of the solution curve.

### Python (Neural ODE)
Requires Python 3 with PyTorch, NumPy, and Matplotlib.

```bash
pip install torch numpy matplotlib
python python/neural_ode.py
```

The script:
1. Generates 1000 training points from the analytical solution (`t = 0` to `5s`)
2. Trains a feedforward network (4 hidden layers: 128, 128, 128, 64 neurons; Tanh activations) for 5000 iterations using Adam and MSE loss
3. Compares the learned solution to the analytical one and prints/plots the prediction at `t = 2s`

## Background: The Model

- **Newton's Second Law:** `F = m·e''(t)`
- **PD Control Force:** `F = -kp·e(t) - kd·e'(t)`
- Combining these gives the governing ODE. The discriminant of its characteristic equation, `D = kd² - 4·m·kp`, determines the robot's behavior:

| Case | Condition | Behavior |
|---|---|---|
| Underdamped | `D < 0` | Robot oscillates across the line |
| Critically damped | `D = 0` | Fastest smooth return (ideal) |
| Overdamped | `D > 0` | Robot drifts back slowly |

## Assumptions & Limitations

- Flat surface, constant forward speed, noise-free sensor readings
- Constant gains (`kp`, `kd`) and mass throughout motion
- PD only (no integral term) — keeps the model second-order
- One-dimensional (lateral error only, no heading/rotational error)
- Valid for small errors (`|e(t)| << 1 m`); nonlinear effects may dominate for larger deviations

## References

See `paper.pdf` for full citations, including Minorsky (1922), Ziegler & Nichols (1942), Chen et al. (2019, *Neural Ordinary Differential Equations*), and Åström & Murray (2002).
