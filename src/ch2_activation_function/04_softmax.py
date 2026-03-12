import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))\

print(softmax(3))
print(softmax(-4))

print(softmax(np.array([0,1])))
print(softmax(np.array([-1,-2])))

print(softmax(np.array([0,0])))