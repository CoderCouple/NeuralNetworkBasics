import numpy as np


# Basic Single Neuron
#
#       Inputs       Weights
#
#    x1 -----(w1)----\
#                     \
#    x2 -----(w2)------\
#                      (N)---> [ sum + b ] ---> z (output)
#    x3 -----(w3)------/|
#                     / |
#    x4 -----(w4)----/  |
#                      bias (b)
#
# Formula: z = x1*w1 + x2*w2 + x3*w3 + x4*w4 + b
#        = dot(x, w) + b

x = [1,2,3,4]
w = [0.3,0.4, 0.9, 0.1]
b = 2

# Single Neron output
z1 = x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + x[3] * w[3] + b
print(z1)

#Matrix Multiplication Output (dot product)
z2 = np.dot(x,w) + b
print(z2)
