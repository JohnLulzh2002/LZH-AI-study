import numpy as np
def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y
def forward(x,w,b):
    a=np.dot(w,x.T)+b
    return sigmoid(a)
X = np.array([1.0, 0.5])
W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B = np.array([0.1, 0.2, 0.3])
print(X)
print()
print(np.dot(X,W))
print()
print(sigmoid(np.dot(X,W)))