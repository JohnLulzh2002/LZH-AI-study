import numpy as np
from CommonFun import sigmoid
X = np.array([1.0, 0.5])
W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B = np.array([0.1, 0.2, 0.3])
print(X)
print(np.dot(X,W))
print(sigmoid(np.dot(X,W)))