import numpy as np

# Mean Squared Error (MSE) Loss Function
#
# Definition:
#   Measures the average of the squared differences between predicted
#   and actual values. The most common loss function for regression tasks.
#
# Formula:
#   MSE = (1/n) * sum((y_true_i - y_pred_i)^2 for i=1..n)
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
#   squared = [0.25, 0.0, 0.25]
#   MSE = (0.25 + 0.0 + 0.25) / 3 = 0.1667
#
# Graph (loss vs prediction for a single sample where y_true = 0):
#
#    Loss |
#         |  \              /
#         |   \            /
#         |    \          /
#         |     \        /
#         |      \      /
#         |       \    /
#         |        \  /
#       0 |_________\/________→ y_pred
#                    0
#                 (y_true)
#
#   The parabola shape means:
#     - Loss is 0 when prediction matches the target exactly
#     - Loss grows quadratically as prediction moves away from target
#     - Large errors are penalized much more than small errors
#
# Derivative (with respect to y_pred):
#   dMSE/dy_pred = (2/n) * (y_pred - y_true)
#
#   This gradient tells us:
#     - Direction: if y_pred > y_true, gradient is positive → decrease prediction
#     - Magnitude: larger errors produce larger gradients → faster correction
#
# When to use:
#   - Regression tasks (predicting continuous values)
#     e.g., house prices, temperature, stock values
#   - When large errors should be penalized heavily
#   - As a default starting point for any regression problem
#
# Advantages:
#   - Differentiable everywhere → works well with gradient descent
#   - Penalizes large errors more due to squaring → model focuses on big mistakes
#   - Unique global minimum → optimization landscape has no local minima
#   - Mathematically convenient and well-understood
#
# Limitations:
#   - Sensitive to outliers: a single large error dominates the total loss
#     e.g., errors [0.1, 0.2, 10.0] → squared [0.01, 0.04, 100.0]
#     The outlier (10.0) contributes 99.95% of the total loss
#   - Not robust in noisy datasets — consider MAE or Huber loss instead
#   - Units are squared (e.g., if predicting meters, MSE is in meters^2)
#     → Use RMSE (sqrt of MSE) for interpretable units
#
# Related loss functions:
#   - MAE (Mean Absolute Error):  less sensitive to outliers, uses |error|
#   - RMSE (Root MSE):            sqrt(MSE), same units as target
#   - Huber Loss:                 MSE for small errors, MAE for large errors


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# --- Single sample ---
# Perfect prediction: loss should be 0
print(mse(np.array([1.0]), np.array([1.0])))

# Off by 1: loss = (1-2)^2 = 1.0
print(mse(np.array([1.0]), np.array([2.0])))

# --- Batch of samples ---
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.5, 5.0, 3.0, 6.0])

# errors:  [0.5, 0.0, -0.5, 1.0]
# squared: [0.25, 0.0, 0.25, 1.0]
# MSE = (0.25 + 0.0 + 0.25 + 1.0) / 4 = 0.375
print(mse(y_true, y_pred))
