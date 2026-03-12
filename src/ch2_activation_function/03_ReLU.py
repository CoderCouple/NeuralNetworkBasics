import numpy as np

def relu(x):
    return np.maximum(0,x)

print(relu(-5))
print(relu(5))
print(relu(np.array([0,0])))
print(relu(np.array([-5,5])))
print(relu(np.array([0,0])))
print(relu(np.array([1,1])))
print(relu(np.array([0,1])))
