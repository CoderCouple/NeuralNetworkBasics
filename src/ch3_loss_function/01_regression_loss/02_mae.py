import numpy as np

# Mean Absolute Error (MAE) Loss Function
#
# Definition:
#   Measures the average of the absolute differences between predicted
#   and actual values. Also known as L1 Loss.
#
# Formula:
#   MAE = (1/n) * sum(|y_true_i - y_pred_i| for i=1..n)
#
#   Where:
#     y_true = actual/target values
#     y_pred = predicted values
#     n      = number of samples
#
# Example:
#   y_true = [3.0, 5.0, 2.5]
#   y_pred = [2.5, 5.0, 3.0]
#   errors = [0.5, 0.0, -0.5]
#   abs    = [0.5, 0.0, 0.5]
#   MAE = (0.5 + 0.0 + 0.5) / 3 = 0.3333
#
# Graph (loss vs prediction for a single sample where y_true = 0):
#
#    Loss |
#         |\              /
#         | \            /
#         |  \          /
#         |   \        /
#         |    \      /
#         |     \    /
#         |      \  /
#       0 |_______\/_________→ y_pred
#                  0
#               (y_true)
#
#   The V-shape means:
#     - Loss is 0 when prediction matches the target exactly
#     - Loss grows linearly as prediction moves away from target
#     - All errors are penalized equally regardless of size
#
# Derivative (with respect to y_pred):
#   dMAE/dy_pred = (1/n) * sign(y_pred - y_true)
#
#   Where sign(x) = +1 if x > 0, -1 if x < 0, undefined at x = 0
#
#   This gradient tells us:
#     - Direction only: gradient is always +1 or -1 (constant magnitude)
#     - Unlike MSE, large and small errors produce the same gradient size
#     - Not differentiable at y_pred == y_true (the sharp corner of the V)
#
# When to use:
#   - Regression tasks where outliers are common or expected
#     e.g., predicting delivery times, sensor readings with noise
#   - When you want the model to treat all errors equally
#   - When the target distribution has heavy tails
#
# Advantages:
#   - Robust to outliers: a single large error does not dominate
#     e.g., errors [0.1, 0.2, 10.0] → abs [0.1, 0.2, 10.0]
#     The outlier contributes 97% (vs 99.95% in MSE)
#   - Same units as the target (easier to interpret than MSE)
#   - Simple and intuitive — "average distance from the truth"
#
# Limitations:
#   - Not differentiable at zero error → gradient is undefined at the
#     exact point we want to reach, can cause instability near convergence
#   - Constant gradient magnitude means the model does not slow down
#     as it approaches the correct answer (can oscillate around minimum)
#   - Converges slower than MSE for well-behaved (outlier-free) data
#
# MAE vs MSE comparison:
#   ┌──────────────┬──────────────────────┬──────────────────────┐
#   │              │       MSE            │       MAE            │
#   ├──────────────┼──────────────────────┼──────────────────────┤
#   │ Penalty      │ Quadratic (x^2)      │ Linear (|x|)         │
#   │ Outliers     │ Sensitive            │ Robust               │
#   │ Gradient     │ Proportional to error│ Constant (±1)        │
#   │ Smoothness   │ Smooth everywhere    │ Sharp corner at 0    │
#   │ Convergence  │ Faster (no outliers) │ Slower but steadier  │
#   │ Units        │ Squared units        │ Same as target       │
#   └──────────────┴──────────────────────┴──────────────────────┘
#
# Related loss functions:
#   - MSE:        penalizes large errors more, smooth gradient
#   - Huber Loss: combines MSE (small errors) and MAE (large errors)
#   - MAPE:       percentage-based, scale-independent


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# --- Single sample ---
# Perfect prediction: loss should be 0
print(mae(np.array([1.0]), np.array([1.0])))

# Off by 1: loss = |1-2| = 1.0
print(mae(np.array([1.0]), np.array([2.0])))

# --- Batch of samples ---
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.5, 5.0, 3.0, 6.0])

# errors:  [0.5, 0.0, -0.5, 1.0]
# abs:     [0.5, 0.0, 0.5, 1.0]
# MAE = (0.5 + 0.0 + 0.5 + 1.0) / 4 = 0.5
print(mae(y_true, y_pred))
