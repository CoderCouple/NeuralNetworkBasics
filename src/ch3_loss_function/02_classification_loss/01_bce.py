import numpy as np

# Binary Cross-Entropy (BCE) Loss Function
#
# Definition:
#   Measures the difference between predicted probabilities and actual
#   binary labels (0 or 1). The standard loss for binary classification.
#   Also called Log Loss.
#
# Formula:
#   BCE = -(1/n) * sum(y_true_i * log(y_pred_i) + (1 - y_true_i) * log(1 - y_pred_i))
#
#   Where:
#     y_true = actual labels (0 or 1)
#     y_pred = predicted probabilities (between 0 and 1, typically from sigmoid)
#     n      = number of samples
#
# How it works (two cases):
#   When y_true = 1:  loss = -log(y_pred)
#     → Penalizes low confidence in the correct class
#     → y_pred = 0.99 → loss = 0.01  (confident & correct → low loss)
#     → y_pred = 0.01 → loss = 4.61  (confident & wrong → high loss)
#
#   When y_true = 0:  loss = -log(1 - y_pred)
#     → Penalizes high confidence in the wrong class
#     → y_pred = 0.01 → loss = 0.01  (confident & correct → low loss)
#     → y_pred = 0.99 → loss = 4.61  (confident & wrong → high loss)
#
# Graph (loss vs y_pred for a single sample):
#
#   When y_true = 1:               When y_true = 0:
#
#   Loss |                          Loss |
#        |                               |\
#        |\                              | \
#        | \                             |  \
#        |  \                            |   \
#        |   \                           |    '.
#        |    '.                         |      '-.
#        |      '-.                      |         '-.__
#        |         '-.__                 |              '---___
#      0 |             '---___         0 |                     '---
#        +-------------------→ y_pred    +-------------------→ y_pred
#        0                   1           0                   1
#
#   Key insight: the log curve creates asymmetric penalties —
#   confident wrong predictions are punished exponentially more
#   than slightly wrong predictions.
#
# Derivative (with respect to y_pred):
#   dBCE/dy_pred = -(1/n) * (y_true/y_pred - (1 - y_true)/(1 - y_pred))
#
#   Simplified:
#   dBCE/dy_pred = -(1/n) * (y_true - y_pred) / (y_pred * (1 - y_pred))
#
#   This gradient:
#     - Points toward the correct answer
#     - Grows very large when prediction is confidently wrong
#     - Approaches zero as prediction gets closer to the truth
#
# When to use:
#   - Binary classification (spam/not spam, cat/dog, disease/healthy)
#   - Multi-label classification (each label is independent binary)
#   - Output layer uses sigmoid activation
#
# Advantages:
#   - Heavily penalizes confident wrong predictions (logarithmic penalty)
#   - Well-calibrated: pushes predictions toward true probabilities
#   - Pairs naturally with sigmoid output → combined gradient is simple
#   - Information-theoretic foundation (measures bits of surprise)
#
# Limitations:
#   - Requires y_pred in (0, 1) — log(0) is undefined (negative infinity)
#     → Fix: clip predictions, e.g., np.clip(y_pred, 1e-15, 1 - 1e-15)
#   - Sensitive to class imbalance (rare class gets drowned out)
#     → Fix: use weighted cross-entropy
#   - Not suitable for multi-class (use categorical cross-entropy instead)
#
# Related loss functions:
#   - Categorical Cross-Entropy: multi-class version (one-hot labels)
#   - Weighted Cross-Entropy: adds class weights for imbalanced data
#   - Focal Loss: down-weights easy examples, focuses on hard ones


def binary_cross_entropy(y_true, y_pred):
    # Clip to avoid log(0) which gives -inf
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# --- Single sample ---
# Correct prediction with high confidence: low loss
print(binary_cross_entropy(np.array([1.0]), np.array([0.99])))

# Wrong prediction with high confidence: high loss
print(binary_cross_entropy(np.array([1.0]), np.array([0.01])))

# --- Batch of samples ---
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7])

# Sample breakdown:
#   y=1, p=0.9 → -log(0.9)     = 0.105
#   y=0, p=0.1 → -log(1 - 0.1) = 0.105
#   y=1, p=0.8 → -log(0.8)     = 0.223
#   y=0, p=0.3 → -log(1 - 0.3) = 0.357
#   y=1, p=0.7 → -log(0.7)     = 0.357
# BCE = mean of above = 0.229
print(binary_cross_entropy(y_true, y_pred))
