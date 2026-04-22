import numpy as np

# Categorical Cross-Entropy (CCE) Loss Function
#
# Definition:
#   Measures the difference between predicted probability distribution
#   and true class labels for multi-class classification.
#   The standard loss when exactly one class is correct.
#
# Formula:
#   For a single sample with K classes:
#   L = -sum(y_true_k * log(y_pred_k) for k=1..K)
#
#   For a batch of n samples:
#   CCE = -(1/n) * sum_i( sum_k( y_true_ik * log(y_pred_ik) ) )
#
#   Where:
#     y_true = one-hot encoded true labels (e.g., [0, 1, 0] for class 2)
#     y_pred = predicted probabilities from softmax (sum to 1)
#     K      = number of classes
#     n      = number of samples
#
# How it simplifies (with one-hot labels):
#   Since y_true is one-hot, only one term survives in the sum:
#   L = -log(y_pred_c)   where c is the true class
#
#   Example: true class = 2, y_true = [0, 1, 0]
#   y_pred = [0.1, 0.7, 0.2]
#   L = -(0*log(0.1) + 1*log(0.7) + 0*log(0.2))
#     = -log(0.7)
#     = 0.357
#
# Graph (loss vs predicted probability of the TRUE class):
#
#    Loss |
#         |\
#         | \
#         |  \
#         |   \
#         |    \
#         |     '.
#         |       '-.
#         |          '-.__
#       0 |               '---___
#         +----------------------------→ p(true class)
#         0         0.5              1
#
#   Loss → infinity as p(true class) → 0
#   Loss → 0 as p(true class) → 1
#
# Derivative (with respect to y_pred, combined with softmax):
#   dL/dz_k = y_pred_k - y_true_k
#
#   Where z is the logit (input to softmax). This remarkably simple
#   gradient is why softmax + cross-entropy are always paired together.
#
# When to use:
#   - Multi-class classification (exactly one correct class)
#     e.g., digit recognition (0-9), animal species, language detection
#   - Output layer uses softmax activation
#   - Labels are one-hot encoded
#
# Advantages:
#   - Natural pairing with softmax → simple, elegant gradient
#   - Heavily penalizes confident wrong predictions
#   - Well-calibrated probability outputs
#   - Information-theoretic foundation (cross-entropy between distributions)
#
# Limitations:
#   - Requires y_pred to sum to 1 (must use softmax output)
#   - log(0) is undefined → clip predictions to avoid numerical issues
#   - Assumes mutually exclusive classes (only one can be true)
#     → For multi-label, use binary cross-entropy per class instead
#   - Sensitive to class imbalance → use weighted version or focal loss
#
# CCE vs BCE:
#   ┌────────────────┬────────────────────────┬────────────────────────┐
#   │                │         BCE            │         CCE            │
#   ├────────────────┼────────────────────────┼────────────────────────┤
#   │ Classes        │ 2 (binary)             │ K (multi-class)        │
#   │ Labels         │ Single value (0 or 1)  │ One-hot vector         │
#   │ Output act.    │ Sigmoid                │ Softmax                │
#   │ Outputs        │ Independent            │ Sum to 1               │
#   │ Use case       │ Spam/not spam          │ Cat/dog/bird           │
#   └────────────────┴────────────────────────┴────────────────────────┘
#
# Related loss functions:
#   - Binary Cross-Entropy: for binary classification
#   - Sparse Categorical CE: same but takes class index instead of one-hot
#   - Focal Loss: focuses on hard-to-classify examples
#   - KL Divergence: generalized version measuring distribution difference


def categorical_cross_entropy(y_true, y_pred):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# --- Single sample (3 classes) ---
# True class is class 2 (index 1), model predicts 0.7 for it → low loss
y_true = np.array([[0, 1, 0]])
y_pred = np.array([[0.1, 0.7, 0.2]])
print(categorical_cross_entropy(y_true, y_pred))  # -log(0.7) = 0.357

# True class is class 2, model predicts 0.1 for it → high loss
y_pred_bad = np.array([[0.7, 0.1, 0.2]])
print(categorical_cross_entropy(y_true, y_pred_bad))  # -log(0.1) = 2.303

# --- Batch of samples (3 classes) ---
y_true = np.array([
    [1, 0, 0],   # true class: 1
    [0, 1, 0],   # true class: 2
    [0, 0, 1],   # true class: 3
])
y_pred = np.array([
    [0.8, 0.1, 0.1],   # good prediction for class 1
    [0.2, 0.6, 0.2],   # decent prediction for class 2
    [0.1, 0.1, 0.8],   # good prediction for class 3
])

# Per-sample loss:
#   -log(0.8) = 0.223
#   -log(0.6) = 0.511
#   -log(0.8) = 0.223
# CCE = (0.223 + 0.511 + 0.223) / 3 = 0.319
print(categorical_cross_entropy(y_true, y_pred))
