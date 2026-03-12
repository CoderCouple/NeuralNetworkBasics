import numpy as np

# Layer of 3 Neurons with 4 Inputs
#
#                          Neuron 1          Neuron 2          Neuron 3
#                         .--------.        .--------.        .--------.
#    x1 ----w11----------( + + b1  )  w12--( + + b2  )  w13--( + + b3  )
#         \   \           '---+----'  /     '---+----'  /     '---+----'
#          \   \              |      /          |      /          |
#    x2 ----+---w21-----------+-----'     w22--'      |    w23--'
#         \  \  \             |          /             |   /
#          \  \  \            |         /              |  /
#    x3 ----+--+--w31--------+--------'         w32--'  |
#         \  \  \  \          |                /         |
#          \  \  \  \         |               /          |
#    x4 ----+--+--+--w41-----+------'  w42--'      w43-'
#            \  \  \
#             \  \  '--- connected to neuron 1
#              \  '------ connected to neuron 2
#               '-------- connected to neuron 3
#
#         Each input xi is connected to every neuron via its own weight wij
#
# Formulas:
#   z1 = x1*w11 + x2*w21 + x3*w31 + x4*w41 + b1  = dot(x, w1) + b1
#   z2 = x1*w12 + x2*w22 + x3*w32 + x4*w42 + b2  = dot(x, w2) + b2
#   z3 = x1*w13 + x2*w23 + x3*w33 + x4*w43 + b3  = dot(x, w3) + b3

x = [1, 2, 3, 4]

w1 = [0.3, 0.4, 0.9, 0.1]
w2 = [0.5, 0.2, 0.7, 0.3]
w3 = [0.8, 0.1, 0.4, 0.6]

w = [w1, w2, w3]

b1 = 2
b2 = 3
b3 = 0.5

b= [b1, b2, b3]

# First Neuron Output = like
z1 = w[0][0] * x[0]  + w[0][1] * x[1] + w[0][2] * x[2] + w[0][3] * x[3] + b1
print(z1)

# Second Neuron Output = np.dot(x, w2) + b2
z2 = w[1][0] * x[0] + w[1][1] * x[1] + w[1][2] * x[2] + w[1][3] * x[3] + b2
print(z2)

# Third Neuron Output = np.dot(x, w3) + b3
z3 = w[2][0] * x[0] + w[2][1] * x[1] + w[2][2] * x[2] + w[2][3] * x[3] + b3
print(z3)

#Matrix Multiplication Output (dot product)
o = np.dot(x,np.asarray(w).T) + b
print(o)
