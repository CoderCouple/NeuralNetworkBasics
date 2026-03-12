import numpy as np

# Softmax Activation Function
#
# Definition:
#   Converts a vector of raw scores (logits) into a probability distribution.
#   Each output is in (0, 1) and all outputs sum to exactly 1.
#
# Formula:
#   For a vector z of length K:
#   f(z_i) = e^(z_i) / sum(e^(z_j) for j=1..K)
#
# Example:
#   z = [2.0, 1.0, 0.1]
#   e^z = [7.39, 2.72, 1.11]
#   sum = 11.22
#   softmax(z) = [0.659, 0.242, 0.099]  → sums to 1.0
#
# Properties:
#   - Output is always a valid probability distribution (sums to 1)
#   - Larger inputs get exponentially more probability
#   - Preserves ranking: if z_i > z_j then softmax(z_i) > softmax(z_j)
#   - Temperature scaling: softmax(z/T) — high T → uniform, low T → one-hot
#
# Derivative:
#   df(z_i)/dz_j = f(z_i) * (delta_ij - f(z_j))
#   where delta_ij = 1 if i==j, else 0
#   This gives a Jacobian matrix, not a simple scalar derivative.
#
# When to use:
#   - Output layer for multi-class classification (exactly one correct class)
#   - The final layer paired with cross-entropy loss
#   - Language models (next token prediction over vocabulary)
#
# Limitations:
#   - Only for output layers, not suitable for hidden layers
#   - Computationally expensive for very large vectors (e.g., large vocabularies)
#   - Numerical instability with large inputs (overflow in e^x)
#     → Fix: subtract max(z) before computing: softmax(z - max(z))
#
# Not to be confused with:
#   - Sigmoid: for binary or multi-label classification (independent outputs)
#   - Softmax: for multi-class classification (mutually exclusive outputs)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

print(softmax(3))
print(softmax(-4))

print(softmax(np.array([0,1])))
print(softmax(np.array([-1,-2])))

print(softmax(np.array([0,0])))