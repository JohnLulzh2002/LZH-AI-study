import numpy as np
import matplotlib.pylab as plt
def step(x):
    y=x>0
    return y.astype(int)
def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y
def ReLU(x):
    return np.maximum(x,0)
x=np.arange(-5,5,0.1)
y=step(x)
plt.plot(x,y,label="step")
y=sigmoid(x)
plt.plot(x,y,label="sigmoid")
y=ReLU(x)
plt.plot(x,y,label="ReLU")
plt.legend()
plt.show()