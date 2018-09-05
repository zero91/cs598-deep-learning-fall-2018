import numpy as np

def relu(z):
    return np.maximum(z, 0)

def gradient_for_relu(z):
    z[z >= 0] = 1
    z[z < 0] = 0
    return z

def sigmoid(z):
    pass

def gradient_for_sigmoid(z):
    pass

def softmax(z):
    # print(np.sum(np.exp(z)))
    return np.exp(z) / np.sum(np.exp(z))