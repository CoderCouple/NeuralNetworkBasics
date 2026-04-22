import numpy as np

# Huber Loss (Smooth Mean Absolute Error)
#
# Definition:
#   A hybrid loss function that behaves like MSE for small errors and
#   like MAE for large errors. Controlled by a threshold parameter delta.
#   Combines the best of both: smooth gradients near zero + outlier robustness.
#
# Formula:
#   For each sample, the error is e = y_true - y_pred:
#
#   L(e) = 0.5 * e^2                  if |e| <= delta
#   L(e) = delta * |e| - 0.5 * delta^2  if |e| > delta
#
#   Huber = (1/n) * sum(L(e_i) for i=1..n)
#
#   Where:
#     y_true = actual/target values
#     y_pred = predicted values
#     delta  = threshold that separates MSE and MAE behavior
#     n      = number of samples
#
# Example (delta = 1.0):
#   y_true = [3.0, 5.0, 2.5, 7.0]
#   y_pred = [2.5, 5.0, 3.0, 2.0]
#   errors = [0.5, 0.0, -0.5, 5.0]
#
#   |0.5| <= 1.0 → 0.5 * 0.5^2            = 0.125   (MSE region)
#   |0.0| <= 1.0 → 0.5 * 0.0^2            = 0.0     (MSE region)
#   |0.5| <= 1.0 → 0.5 * 0.5^2            = 0.125   (MSE region)
#   |5.0| >  1.0 → 1.0 * 5.0 - 0.5 * 1.0  = 4.5     (MAE region)
#
#   Huber = (0.125 + 0.0 + 0.125 + 4.5) / 4 = 1.1875
#
# Graph (loss vs error, delta = 1.0):
#
#    Loss |
#         |  \                    /      ← MAE region (linear)
#         |   \                  /
#         |    \                /
#         |     \              /
#         |      \            /
#         |       \..      ../           ← MSE region (quadratic)
#         |          ''--''
#       0 |____________|_____________→ error
#                      0
#               -delta   +delta
#               ←─────→ ←─────→
#                MSE       MSE
#          MAE ←              → MAE
#
#   Key visual insight:
#     - Near zero: smooth parabola (like MSE) → stable gradients
#     - Far from zero: straight lines (like MAE) → bounded gradients
#     - The transition at ±delta is smooth (no sharp corners)
#
# Derivative (with respect to y_pred):
#   dL/dy_pred = -(y_true - y_pred)           if |e| <= delta  (same as MSE)
#   dL/dy_pred = -delta * sign(y_true - y_pred)  if |e| > delta   (same as MAE)
#
#   This means:
#     - Small errors: gradient is proportional to error → precise convergence
#     - Large errors: gradient is capped at delta → no exploding gradients
#     - Smooth everywhere including at ±delta → stable training
#
# The delta parameter:
#   delta controls where the transition from MSE to MAE happens.
#
#   ┌──────────────┬──────────────────────────────────────────────┐
#   │ delta value  │ Behavior                                     │
#   ├──────────────┼──────────────────────────────────────────────┤
#   │ Very large   │ Approaches MSE (everything is "small error") │
#   │ delta = 1.0  │ Common default, balanced behavior            │
#   │ Very small   │ Approaches MAE (everything is "large error") │
#   └──────────────┴──────────────────────────────────────────────┘
#
# When to use:
#   - Regression tasks with some outliers but not extremely noisy data
#   - When MSE causes exploding gradients due to outliers
#   - When MAE converges too slowly due to constant gradients
#   - Robotics, reinforcement learning (e.g., DQN uses Huber loss)
#
# Advantages:
#   - Best of both worlds: MSE precision + MAE robustness
#   - Differentiable everywhere (including at ±delta) → smooth training
#   - Bounded gradients → no exploding gradient from outliers
#   - Tunable via delta → adapt to your data's noise level
#
# Limitations:
#   - Extra hyperparameter (delta) that requires tuning
#   - Slightly more complex to implement than MSE or MAE
#   - If delta is poorly chosen, can behave like pure MSE or pure MAE
#
# Comparison with MSE and MAE:
#   ┌──────────────┬──────────────┬──────────────┬──────────────────┐
#   │              │     MSE      │     MAE      │    Huber         │
#   ├──────────────┼──────────────┼──────────────┼──────────────────┤
#   │ Small errors │ Quadratic    │ Linear       │ Quadratic (MSE)  │
#   │ Large errors │ Quadratic    │ Linear       │ Linear (MAE)     │
#   │ Outliers     │ Sensitive    │ Robust       │ Robust           │
#   │ Smoothness   │ Smooth       │ Sharp at 0   │ Smooth           │
#   │ Gradient     │ Proportional │ Constant ±1  │ Proportional/Cap │
#   │ Convergence  │ Fast         │ Slow         │ Fast + stable    │
#   └──────────────┴──────────────┴──────────────┴──────────────────┘


def huber(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return np.mean(np.where(is_small, squared_loss, linear_loss))


# --- Single sample ---
# Perfect prediction: loss should be 0
print(huber(np.array([1.0]), np.array([1.0])))

# Small error (|e| = 0.5 <= delta=1.0): uses MSE region → 0.5 * 0.5^2 = 0.125
print(huber(np.array([1.0]), np.array([1.5])))

# Large error (|e| = 5.0 > delta=1.0): uses MAE region → 1.0 * 5.0 - 0.5 = 4.5
print(huber(np.array([1.0]), np.array([6.0])))

# --- Batch of samples ---
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.5, 5.0, 3.0, 2.0])

# errors:  [0.5, 0.0, -0.5, 5.0]
# losses:  [0.125, 0.0, 0.125, 4.5]  (first 3 in MSE region, last in MAE region)
# Huber = (0.125 + 0.0 + 0.125 + 4.5) / 4 = 1.1875
print(huber(y_true, y_pred))

# --- Effect of delta ---
# Same data, different delta values
print(huber(y_true, y_pred, delta=0.1))   # small delta → more MAE-like
print(huber(y_true, y_pred, delta=10.0))  # large delta → more MSE-like
