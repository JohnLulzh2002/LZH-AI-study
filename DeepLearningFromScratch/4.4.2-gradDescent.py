import numpy as np
from CommonFun import softmax,cross_entropy_error,numerical_gradient
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net=simpleNet()
print('W=',net.W)
x=np.array([0.6, 0.9])
print('x=',x,'\tp(x)=',net.predict(x),sep='')
t=np.array([0,0,1])
f=lambda nothing: net.loss(x,t)
print('loss=',f(None),sep='')
dW=numerical_gradient(f,net.W)
print('dW=',dW,end='\n\n')

def grad_descent(f,x,lr=0.05,rep=2000):
    for i in range(rep):
        x-=numerical_gradient(f,x)*lr
    return x
net.W=grad_descent(f,net.W)
print('p(x)=',net.predict(x))
print('loss=',f(None),sep='')