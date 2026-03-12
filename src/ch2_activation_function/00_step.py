# Step Activation Function
#
# Definition:
#   The simplest activation function. It outputs a binary value (0 or 1)
#   based on whether the input exceeds a threshold.
#
# Formula:
#   f(x) = 1  if x > threshold
#   f(x) = 0  if x < threshold
#
# Graph:
#        1 |         ________
#          |        |
#          |        |
#        0 |________|
#          +--------+--------→ x
#               threshold
#
# Derivative:
#   f'(x) = 0 everywhere (except at threshold, where it is undefined)
#   Because the derivative is 0, gradient-based learning cannot work.
#
# When to use:
#   - Binary classification in simple perceptrons (historical use)
#   - Biological neuron modeling (fire or not fire)
#
# Limitations:
#   - Not differentiable at the threshold → cannot use with backpropagation
#   - No gradient signal → weights cannot learn via gradient descent
#   - Cannot handle multi-class or probabilistic outputs
#   - Largely replaced by sigmoid, ReLU, and softmax in modern networks

def step(x , threshold):
    if x < threshold:
        return 0
    elif x> threshold:
        return 1;
    else:
        return None


print(step(5, 1))
print(step(4, 1))
print(step(-2, 1))
print(step(1, 1))
print(step(0, 1))
