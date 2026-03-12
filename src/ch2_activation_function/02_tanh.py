import numpy as np

# Tanh (Hyperbolic Tangent) Activation Function
#
# Definition:
#   Squashes any real-valued input into the range (-1, 1).
#   Similar to sigmoid but zero-centered.
#
# Formula:
#   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
#
# Relationship to sigmoid:
#   tanh(x) = 2 * sigmoid(2x) - 1
#
# Graph:
#      1.0 |                 ___-------
#          |              /
#      0.0 |- - - - - -/- - - - - - -
#          |          /
#     -1.0 |___------
#          +-------------------------→ x
#
# Derivative:
#   f'(x) = 1 - f(x)^2
#   Maximum gradient at x=0 (1.0), stronger than sigmoid's max (0.25).
#
# When to use:
#   - Hidden layers when zero-centered output is needed
#   - RNNs (recurrent neural networks) for hidden state computation
#   - When inputs have both positive and negative values
#
# Advantages over sigmoid:
#   - Zero-centered output → faster convergence in practice
#   - Stronger gradients (max 1.0 vs 0.25)
#
# Limitations:
#   - Still suffers from vanishing gradient for large |x|
#   - Computationally more expensive than ReLU

def tanh(x):
    return np.tanh(x)

print(tanh(np.array([0,0])))
print(tanh(np.array([1,1])))
print(tanh(np.array([0,0])))