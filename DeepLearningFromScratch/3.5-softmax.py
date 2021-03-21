import numpy as np
def softmax(x):
    m=np.max(x)
    exp_x=np.exp(x-m)
    s=np.sum(exp_x)
    return exp_x/s
a=np.array([0.3,2.9,4.0])
b=np.array([1010,1000,990])
print('a=',a)
print('softmax(a)=',softmax(a))
print('softmax(b)=',softmax(b))