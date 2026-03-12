import numpy as np

# Batch of Inputs through a Layer of 3 Neurons
#
# A batch contains multiple samples, each with 4 features (x1, x2, x3, x4).
# Each sample passes through the same 3 neurons with shared weights and biases.
#
#  Batch (3 samples):       Weights (3 neurons x 4 inputs):     Biases:
#  ┌─────────────────┐      ┌─────────────────────────┐        ┌────────┐
#  │ x1  x2  x3  x4  │      │ w11  w12  w13  w14 │ N1 │        │  b1    │
#  │ sample 1        │      │ w21  w22  w23  w24 │ N2 │        │  b2    │
#  │ sample 2        │      │ w31  w32  w33  w34 │ N3 │        │  b3    │
#  │ sample 3        │      └─────────────────────────┘        └────────┘
#  └─────────────────┘
#
# Formula:  Z = X . W^T + b
#
#   Where:
#     X   = (batch_size x 4) matrix of inputs
#     W^T = (4 x 3) transposed weight matrix
#     b   = (1 x 3) bias vector (broadcast across batch)
#     Z   = (batch_size x 3) output matrix


# Batch of 3 samples, each with 4 inputs
X = np.array([
    [1.0, 2.0, 3.0, 4.0],   # sample 1
    [5.0, 6.0, 7.0, 8.0],   # sample 2
    [9.0, 10.0, 11.0, 12.0]  # sample 3
])

# Weights for 3 neurons (each neuron has 4 weights)
w1 = [0.3, 0.4, 0.9, 0.1]
w2 = [0.5, 0.2, 0.7, 0.3]
w3 = [0.8, 0.1, 0.4, 0.6]

W = np.array([w1, w2, w3])

# Biases for 3 neurons
b = np.array([2.0, 3.0, 0.5])

# Batch output using matrix multiplication
#
# Z = X . W^T + b  (3x4 . 4x3 = 3x3) + (1x3 broadcast)
#
#              Neuron1 (w1)        Neuron2 (w2)        Neuron3 (w3)
#            ┌──────────────────┬──────────────────┬──────────────────┐
# sample 1   │ np.dot(X[0],w1)  │ np.dot(X[0],w2)  │ np.dot(X[0],w3)  │
#            │   + b1           │   + b2           │   + b3           │
#            ├──────────────────┼──────────────────┼──────────────────┤
# sample 2   │ np.dot(X[1],w1)  │ np.dot(X[1],w2)  │ np.dot(X[1],w3)  │
#            │   + b1           │   + b2           │   + b3           │
#            ├──────────────────┼──────────────────┼──────────────────┤
# sample 3   │ np.dot(X[2],w1)  │ np.dot(X[2],w2)  │ np.dot(X[2],w3)  │
#            │   + b1           │   + b2           │   + b3           │
#            └──────────────────┴──────────────────┴──────────────────┘
#
Z = np.dot( X, W.T) + b
print(Z)
