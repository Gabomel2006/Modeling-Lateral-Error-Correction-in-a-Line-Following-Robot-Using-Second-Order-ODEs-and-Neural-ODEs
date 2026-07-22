# ================================================================
# Neural ODE Verification — Line Following Robot
# Gabriel Melgarejo — Math 2A
# ================================================================
# New approach: Train network directly on (t, e(t)) pairs
# instead of using RK4 integration internally.
# This is more stable for oscillating solutions.
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------
# SAVE PATH
# ----------------------------------------------------------------
save_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'neural_ode.png')

# ----------------------------------------------------------------
# STEP 1 — Analytical Solution
# ----------------------------------------------------------------
def analytical_solution(t):
    alpha = -1.0
    beta  = np.sqrt(3)
    C1    = 0.1
    C2    = 0.1 / np.sqrt(3)
    return np.exp(alpha * t) * (
        C1 * np.cos(beta * t) +
        C2 * np.sin(beta * t))

# Generate training data from analytical solution
t_max    = 5.0
t_size   = 1000
t_np     = np.linspace(0, t_max, t_size)
e_np     = analytical_solution(t_np)

# Convert to tensors
t_tensor = torch.tensor(
    t_np, dtype=torch.float32).view(-1, 1)
e_tensor = torch.tensor(
    e_np, dtype=torch.float32).view(-1, 1)

print(f"Training data points: {t_size}")
print(f"Time range: 0 to {t_max} seconds")
print(f"Initial error: {e_np[0]:.4f} meters")

# ----------------------------------------------------------------
# STEP 2 — Neural Network
# New approach: Network learns e(t) directly as a function of t
# Input: t (time)
# Output: e(t) (lateral error)
# This avoids RK4 instability completely
# ----------------------------------------------------------------
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.net(t)

# ----------------------------------------------------------------
# STEP 3 — Training
# ----------------------------------------------------------------
model     = NeuralODE()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=200, min_lr=1e-6)
n_iters   = 5000

print("\nTraining Neural ODE...")
print("-" * 50)

best_loss  = float('inf')
best_state = None

for i in range(n_iters):
    optimizer.zero_grad()

    # Forward pass
    pred = model(t_tensor)

    # Loss: MSE between prediction and analytical solution
    loss = torch.mean((pred - e_tensor) ** 2)

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step(loss)

    if loss.item() < best_loss:
        best_loss  = loss.item()
        best_state = {k: v.clone()
            for k, v in model.state_dict().items()}

    if i % 500 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Iter {i:04d} | "
              f"Loss {loss.item():.8f} | "
              f"LR {lr:.6f}")

# Load best model
model.load_state_dict(best_state)

print("-" * 50)
print(f"Training Complete!")
print(f"Best loss: {best_loss:.8f}")

# ----------------------------------------------------------------
# STEP 4 — Generate predictions
# ----------------------------------------------------------------
with torch.no_grad():
    pred_tensor = model(t_tensor)
pred_np = pred_tensor.numpy().flatten()

# ----------------------------------------------------------------
# STEP 5 — Prediction at t = 2 seconds
# ----------------------------------------------------------------
t_pred_val  = 2.0
e_pred_anal = analytical_solution(t_pred_val)

# Neural network prediction at t=2
t2_tensor   = torch.tensor(
    [[t_pred_val]], dtype=torch.float32)
with torch.no_grad():
    e_pred_neur = model(t2_tensor).item()

print(f"\n--- Prediction at t = {t_pred_val} seconds ---")
print(f"Analytical:  e(2) = {e_pred_anal:.6f} meters")
print(f"Neural ODE:  e(2) = {e_pred_neur:.6f} meters")
print(f"Difference:  "
      f"{abs(e_pred_anal - e_pred_neur):.6f} meters")

# ----------------------------------------------------------------
# STEP 6 — Plot
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Analytical solution
ax.plot(t_np, e_np, 'g-', linewidth=4,
        alpha=0.5,
        label='Analytical Solution '
              '(Characteristic Equation)')

# Neural ODE solution
ax.plot(t_np, pred_np, 'k--', linewidth=2,
        label='Neural ODE (Learned from Data)')

# Target line
ax.axhline(y=0, color='r', linestyle=':',
           linewidth=1.5,
           label='Target: $e(t) = 0$')

# Prediction point
ax.plot(t_pred_val, e_pred_anal, 'yo',
        markersize=12,
        markeredgecolor='black',
        label=f'Prediction: $e(2) \\approx'
              f' {e_pred_anal:.4f}$ m')

ax.set_xlabel('Time $t$ (seconds)', fontsize=13)
ax.set_ylabel('Lateral Error $e(t)$ (meters)',
              fontsize=13)
ax.set_title(
    'Neural ODE Learning Robot Error Correction\n'
    '(Characteristic Equation vs Neural ODE)',
    fontsize=14)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, t_max])

plt.tight_layout()
plt.savefig(save_path, dpi=300,
            bbox_inches='tight')
print(f"\nGraph saved to: {save_path}")
plt.show()