# Linear (Identity) Activation Function
#
# Definition:
#   Passes the input directly to the output without any transformation.
#   Also called the identity function — the neuron's output equals its input.
#
# Formula:
#   f(x) = x
#
# Graph:
#          |        /
#          |      /
#          |    /
#          |  /
#          |/
#         /|
#       /  |
#     /    |
#          +-------------------------→ x
#
# Derivative:
#   f'(x) = 1  (constant everywhere)
#
# When to use:
#   - Output layer for regression tasks (predicting continuous values)
#     e.g., predicting house prices, temperature, stock values
#   - Sometimes in the final layer when output can be any real number
#
# Limitations:
#   - No non-linearity → stacking multiple linear layers collapses into
#     a single linear transformation: W2(W1*x + b1) + b2 = W*x + b
#     So a deep network of linear layers is equivalent to one layer.
#   - Cannot learn non-linear patterns (XOR, curves, decision boundaries)
#   - Never use in hidden layers — defeats the purpose of deep learning
#   - Gradient is constant (1), so no vanishing gradient, but also no
#     ability to model complex functions

def linear(x):
    return x

print(linear(2))
print(linear(-2))