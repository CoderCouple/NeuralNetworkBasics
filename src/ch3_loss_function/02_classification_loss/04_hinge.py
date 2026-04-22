import numpy as np

# Hinge Loss Function
#
# Definition:
#   Measures the margin between the correct class score and the decision
#   boundary. Originally designed for Support Vector Machines (SVMs),
#   also used in some neural networks.
#
# Formula (binary):
#   L = (1/n) * sum( max(0, 1 - y_true_i * y_pred_i) )
#
#   Where:
#     y_true = actual labels (-1 or +1)  ← note: NOT 0/1
#     y_pred = raw model output (score, not probability)
#     n      = number of samples
#
# How it works:
#   The product y_true * y_pred tells us if prediction is correct:
#     y_true * y_pred > 0  → correct side of decision boundary
#     y_true * y_pred < 0  → wrong side
#     y_true * y_pred = 1  → exactly on the margin
#
#   Hinge loss penalizes:
#     y_true * y_pred >= 1  → loss = 0       (correct & confident → no penalty)
#     0 < y_true * y_pred < 1 → loss > 0     (correct but not confident enough)
#     y_true * y_pred < 0   → loss > 1       (wrong prediction → big penalty)
#
# Example:
#   y_true = +1, y_pred = 2.0  → max(0, 1 - 1*2.0)  = max(0, -1)  = 0   (safe)
#   y_true = +1, y_pred = 0.5  → max(0, 1 - 1*0.5)  = max(0, 0.5) = 0.5 (margin)
#   y_true = +1, y_pred = -1.0 → max(0, 1 - 1*(-1)) = max(0, 2.0) = 2.0 (wrong)
#   y_true = -1, y_pred = -2.0 → max(0, 1 - (-1)*(-2)) = max(0,-1) = 0  (safe)
#
# Graph (loss vs y_pred for y_true = +1):
#
#    Loss |
#       3 |\
#         | \
#       2 |  \
#         |   \
#       1 |    \
#         |     \
#       0 |______\_______________→ y_pred
#        -2  -1   0   1   2   3
#                  margin
#
#   Key features:
#     - Linear penalty for wrong predictions (like MAE)
#     - Loss reaches 0 at y_pred = 1 (the margin)
#     - No penalty at all beyond the margin → sparse gradients
#     - The flat region means correctly classified samples don't
#       contribute to the gradient at all
#
# Derivative (with respect to y_pred):
#   dL/dy_pred = -y_true   if y_true * y_pred < 1
#   dL/dy_pred = 0         if y_true * y_pred >= 1
#
#   This means:
#     - Samples within the margin or misclassified: gradient pushes
#       prediction toward the correct side
#     - Samples beyond the margin: zero gradient → no update needed
#     - This sparsity makes training efficient (only hard examples contribute)
#
# When to use:
#   - Support Vector Machines (SVMs) — its original and primary use
#   - When you want a max-margin classifier
#   - When you care about decision boundary quality, not probabilities
#   - Some adversarial training setups in neural networks
#
# Advantages:
#   - Encourages a margin of safety between classes (robust boundary)
#   - Sparse gradients: only misclassified or borderline samples update
#     the model → computationally efficient
#   - Robust to outliers far from the boundary (they get linear, not
#     exponential penalty)
#   - Does not require probability calibration
#
# Limitations:
#   - Requires labels to be -1/+1, not 0/1 (needs conversion)
#   - Does not output probabilities (unlike cross-entropy)
#   - Not differentiable at y_true * y_pred = 1 (the kink)
#   - Less common in deep learning — cross-entropy dominates
#   - Multi-class extension is more complex
#
# Hinge vs Cross-Entropy:
#   ┌──────────────────┬──────────────────────┬──────────────────────┐
#   │                  │    Hinge Loss         │  Cross-Entropy       │
#   ├──────────────────┼──────────────────────┼──────────────────────┤
#   │ Labels           │ -1 / +1              │ 0 / 1                │
#   │ Output           │ Raw score            │ Probability          │
#   │ Penalty type     │ Linear               │ Logarithmic          │
#   │ Zero loss at     │ Margin (score >= 1)   │ Never (approaches 0) │
#   │ Gradient         │ Sparse (0 beyond     │ Always non-zero      │
#   │                  │ margin)              │                      │
#   │ Primary use      │ SVMs                 │ Neural networks      │
#   │ Calibration      │ No probabilities     │ Well-calibrated      │
#   └──────────────────┴──────────────────────┴──────────────────────┘
#
# Variants:
#   - Squared Hinge: L = max(0, 1 - y*f)^2  → differentiable at margin
#   - Multi-class Hinge: L = sum( max(0, f_j - f_c + 1) ) for j != c
#     where c is the correct class


def hinge_loss(y_true, y_pred):
    # y_true must be -1 or +1
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


# --- Single samples ---
# Correct and confident: loss = 0
print(hinge_loss(np.array([1]), np.array([2.0])))

# Correct but within margin: loss = 0.5
print(hinge_loss(np.array([1]), np.array([0.5])))

# Wrong prediction: loss = 2.0
print(hinge_loss(np.array([1]), np.array([-1.0])))

# --- Batch of samples ---
y_true = np.array([1, -1, 1, -1, 1])
y_pred = np.array([2.0, -1.5, 0.5, 0.2, -0.5])

# Per-sample:
#   (+1)*2.0  = 2.0  → max(0, 1-2.0)  = 0.0  (correct, beyond margin)
#   (-1)*-1.5 = 1.5  → max(0, 1-1.5)  = 0.0  (correct, beyond margin)
#   (+1)*0.5  = 0.5  → max(0, 1-0.5)  = 0.5  (correct, within margin)
#   (-1)*0.2  = -0.2 → max(0, 1+0.2)  = 1.2  (wrong side!)
#   (+1)*-0.5 = -0.5 → max(0, 1+0.5)  = 1.5  (wrong side!)
#
# Hinge = (0.0 + 0.0 + 0.5 + 1.2 + 1.5) / 5 = 0.64
print(hinge_loss(y_true, y_pred))
