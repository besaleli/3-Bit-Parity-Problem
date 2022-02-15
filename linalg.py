import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def linear_transform(X, weights, bias):
    return np.dot(X, weights) + bias