import numpy as np
def f1(x):
    return x[0]*x[0]+x[1]*x[1]
def numerical_grad(f,x):
    x=x.astype(float)
    h=1e-4
    ans=np.array([])
    for i in range(x.size):
        t=x[i]
        x[i]=t+h
        # print('^',x[i],t+h)
        fx1=f(x)
        x[i]=t-h
        # print('^',x[i],t-h)
        fx2=f(x)
        x[i]=t
        ans=np.append(ans,(fx1-fx2)/(2*h))
    return ans

def grad_output(x):
    print('x=',x,'\tf(x)=',f1(x),'\tgrad=',numerical_grad(f1,x))
grad_output(np.array([3,4]))
grad_output(np.array([0,2]))
grad_output(np.array([3,0]))
print()

def grad_descent(f,x,lr=0.01,rep=100):
    x=x.astype(float)
    for i in range(rep):
        x-=numerical_grad(f,x)*lr
    return x
x=np.array([3,4])
print('x=',x,'\tf(x)=',f1(x))
x=grad_descent(f1,x,lr=0.5,rep=10000)
print('x=',x,'\tf(x)=',f1(x))