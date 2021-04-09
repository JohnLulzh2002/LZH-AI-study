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

def cross_entropy_error(y, t, one_hot=False):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    d = 1e-8
    if one_hot:
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    else:
        return -np.sum(t * np.log(y + d))/batch_size

def numerical_grad(f,x):
    # x=x.astype(float)
    h=1e-4
    ans=np.zeros_like(x)
    if x.ndim==1:
        for i in range(x.size):
            t=x[i]
            x[i]=float(t)+h
            fx1=f(x)
            x[i]=float(t)-h
            fx2=f(x)
            x[i]=t
            ans[i]=(fx1-fx2)/(2*h)
        return ans
    for i in range(len(x)):
        for j in range(len(x[i])):
            t=x[i][j]
            x[i][j]=float(t)+h
            fx1=f(x)
            x[i][j]=float(t)-h
            fx2=f(x)
            x[i][j]=t
            ans[i][j]=(fx1-fx2)/(2*h)
    return ans