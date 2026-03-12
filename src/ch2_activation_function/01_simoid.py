import numpy as np

# Sigmoid Activation Function
#
# Definition:
#   Squashes any real-valued input into the range (0, 1).
#   Outputs can be interpreted as probabilities.
#
# Formula:
#   f(x) = 1 / (1 + e^(-x))
#
# Graph:
#      1.0 |                 ___-------
#          |              /
#      0.5 |            /
#          |          /
#      0.0 |___------
#          +-------------------------→ x
#
# Derivative:
#   f'(x) = f(x) * (1 - f(x))
#   Maximum gradient at x=0 (0.25), vanishes as |x| grows large.
#
# When to use:
#   - Output layer for binary classification (outputs a probability)
#   - Gates in LSTM and GRU recurrent networks
#
# Limitations:
#   - Vanishing gradient problem: for very large or small x, gradient ≈ 0,
#     making deep networks hard to train
#   - Outputs are not zero-centered (all positive), which can slow
#     convergence during gradient descent
#   - Computationally more expensive than ReLU (due to exp)
#   - Largely replaced by ReLU in hidden layers of modern networks

def sigmoid(x):
    return 1/ 1 + np.exp(-x)


print(sigmoid(2))
print(sigmoid(-2))
