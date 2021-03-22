import numpy as np


def ReLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_function(x):
    return .5 * (1 + np.tanh(.5 * x))


def forward(x, w, b):
    a = np.dot(x, w) + b
    # return sigmoid(a)
    return logistic_function(a)


def softmax(x):
    m = np.max(x)
    exp_x = np.exp(x - m)
    s = np.sum(exp_x)
    return exp_x / s