import numpy as np

# ReLU (Rectified Linear Unit) Activation Function
#
# Definition:
#   Outputs the input directly if positive, otherwise outputs zero.
#   The most widely used activation function in modern deep learning.
#
# Formula:
#   f(x) = max(0, x)
#
# Graph:
#         Y|        /
#          |      /
#          |    /
#          |  /
#      0   |/
#          +-------------------------→ x
#
# Derivative:
#   f'(x) = 1  if x > 0
#   f'(x) = 0  if x < 0
#   (undefined at x=0, typically set to 0 in practice)
#
# When to use:
#   - Default choice for hidden layers in most neural networks
#   - CNNs (convolutional neural networks)
#   - Feedforward networks
#   - Any deep network where training speed matters
#
# Advantages:
#   - No vanishing gradient for positive inputs (gradient is always 1)
#   - Computationally very cheap (just a threshold at zero)
#   - Leads to sparse activations (some neurons output 0) → efficient
#   - Converges much faster than sigmoid/tanh in practice
#
# Limitations:
#   - Dying ReLU problem: if a neuron's input is always negative,
#     gradient is always 0 and the neuron stops learning permanently
#   - Not zero-centered (outputs are always >= 0)
#   - Unbounded output (can grow very large)
#
# Variants that address dying ReLU:
#   - Leaky ReLU:  f(x) = max(0.01*x, x)
#   - Parametric ReLU (PReLU): f(x) = max(alpha*x, x), alpha is learned
#   - ELU: f(x) = x if x>0, alpha*(e^x - 1) if x<=0

def relu(x):
    return np.maximum(0,x)

print(relu(-5))
print(relu(5))
print(relu(np.array([0,0])))
print(relu(np.array([-5,5])))
print(relu(np.array([0,0])))
print(relu(np.array([1,1])))
print(relu(np.array([0,1])))
