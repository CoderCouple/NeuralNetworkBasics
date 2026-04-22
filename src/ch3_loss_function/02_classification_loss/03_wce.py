import numpy as np

# Weighted Cross-Entropy (WCE) Loss Function
#
# Definition:
#   A variant of cross-entropy that assigns different weights to each
#   class. Designed to handle class imbalance, where some classes have
#   far fewer samples than others.
#
# The Problem It Solves:
#   In imbalanced datasets, standard cross-entropy treats all classes
#   equally. If 95% of samples are class A and 5% are class B, the
#   model can achieve 95% accuracy by always predicting class A —
#   while completely ignoring class B.
#
#   Example - disease detection:
#     Dataset: 950 healthy, 50 diseased
#     Standard CE → model learns to always predict "healthy"
#     Weighted CE → higher penalty for missing disease cases
#
# Formula (binary weighted cross-entropy):
#   WCE = -(1/n) * sum( w1 * y_true * log(y_pred) + w0 * (1-y_true) * log(1-y_pred) )
#
#   Where:
#     w1 = weight for positive class (class 1)
#     w0 = weight for negative class (class 0)
#     y_true = actual labels (0 or 1)
#     y_pred = predicted probabilities
#
# Formula (multi-class weighted cross-entropy):
#   WCE = -(1/n) * sum_i( sum_k( w_k * y_true_ik * log(y_pred_ik) ) )
#
#   Where:
#     w_k = weight for class k
#
# How to choose weights:
#   Common strategy — inverse frequency:
#     w_k = total_samples / (num_classes * count_of_class_k)
#
#   Example:
#     Classes:  [cat, dog, bird]
#     Counts:   [500, 300, 200]    total = 1000
#     Weights:  [1000/(3*500), 1000/(3*300), 1000/(3*200)]
#             = [0.67, 1.11, 1.67]
#
#     Bird (rarest) gets the highest weight → model pays more
#     attention to getting bird samples correct.
#
# Graph (effect of weights on loss):
#
#   Loss |
#        |  w=5.0 \
#        |         \      w=1.0
#        |          \    /
#        |    w=5.0  \  / w=1.0
#        |        '.  \/ /
#        |          '-.'/
#        |            /'-.
#        |           /    '-.__
#      0 |          /          '---
#        +--------------------------→ p(true class)
#        0                        1
#
#   Higher weight → steeper loss curve → stronger penalty for errors
#
# Derivative (binary, with respect to y_pred):
#   dWCE/dy_pred = -(1/n) * (w1 * y_true/y_pred - w0 * (1-y_true)/(1-y_pred))
#
#   The weight scales the gradient, making the model correct harder
#   on high-weight (minority) classes.
#
# When to use:
#   - Imbalanced datasets (fraud detection, disease diagnosis, rare events)
#   - When false negatives for the minority class are costly
#   - Any classification where some errors matter more than others
#
# Advantages:
#   - Directly addresses class imbalance without resampling data
#   - Easy to implement — just multiply standard CE by weights
#   - Flexible: weights can encode domain knowledge about error costs
#     e.g., missing cancer is worse than a false alarm
#
# Limitations:
#   - Requires choosing weights — poor choices can hurt performance
#   - Does not help if the minority class is inherently hard to learn
#   - May cause the model to over-predict the minority class
#   - For extreme imbalance, consider focal loss or data resampling
#
# Related loss functions:
#   - Binary Cross-Entropy: unweighted, treats classes equally
#   - Focal Loss: automatically down-weights easy examples
#   - Dice Loss: overlap-based, popular in segmentation tasks


def weighted_binary_cross_entropy(y_true, y_pred, w_pos, w_neg):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(
        w_pos * y_true * np.log(y_pred) +
        w_neg * (1 - y_true) * np.log(1 - y_pred)
    )


def weighted_categorical_cross_entropy(y_true, y_pred, weights):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # weights shape: (K,) — one weight per class
    # y_true shape: (n, K), y_pred shape: (n, K)
    return -np.mean(np.sum(weights * y_true * np.log(y_pred), axis=1))


# ============================
# Binary Weighted Cross-Entropy
# ============================

# Imbalanced dataset: 95% negative, 5% positive
# Weight the positive class 10x more to compensate
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
y_pred = np.array([0.1, 0.2, 0.1, 0.05, 0.1, 0.15, 0.1, 0.2, 0.1, 0.8])

# Unweighted (standard BCE) — missing the rare positive costs little
print("Unweighted BCE:")
y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
print(-np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)))

# Weighted — missing the rare positive now costs much more
print("Weighted BCE (w_pos=10, w_neg=1):")
print(weighted_binary_cross_entropy(y_true, y_pred, w_pos=10.0, w_neg=1.0))

# ====================================
# Multi-class Weighted Cross-Entropy
# ====================================

# 3 classes with imbalance: cat(500), dog(300), bird(200)
# Inverse frequency weights
weights = np.array([0.67, 1.11, 1.67])

y_true = np.array([
    [1, 0, 0],   # cat
    [0, 1, 0],   # dog
    [0, 0, 1],   # bird (rarest)
])
y_pred = np.array([
    [0.8, 0.1, 0.1],   # good on cat
    [0.2, 0.6, 0.2],   # decent on dog
    [0.3, 0.3, 0.4],   # poor on bird
])

# Without weights: all classes treated equally
print("\nUnweighted CCE:")
y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
print(-np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1)))

# With weights: bird errors penalized more (w=1.67)
print("Weighted CCE:")
print(weighted_categorical_cross_entropy(y_true, y_pred, weights))
