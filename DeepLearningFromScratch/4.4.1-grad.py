import numpy as np
def numerical_grad(f,x):
    x=x.astype(float)
    h=1e-4
    ans=np.zeros_like(x)
    if x.ndim==1:
        for i in range(x.size):
            t=x[i]
            x[i]=t+h
            fx1=f(x)
            x[i]=t-h
            fx2=f(x)
            x[i]=t
            ans[i]=(fx1-fx2)/(2*h)
        return ans
    for i in range(len(x)):
        for j in range(len(x[i])):
            t=x[i][j]
            x[i][j]=t+h
            fx1=f(x)
            x[i][j]=t-h
            fx2=f(x)
            x[i][j]=t
            ans[i][j]=(fx1-fx2)/(2*h)
    return ans

def grad_output(f,x):
    print('x=',x,'\tf(x)=',f(x),'\tgrad=',numerical_grad(f,x))

def f1(x):
    return x[0]*x[0]+x[1]*x[1]
grad_output(f1,np.array([3,4]))
grad_output(f1,np.array([0,2]))
grad_output(f1,np.array([3,0]))
print()

def f2(x):
    return x[0][0]*x[0][0]+x[0][1]*x[0][1]+x[1][0]*x[1][0]+x[1][1]*x[1][1]+x[2][0]*x[2][0]+x[2][1]*x[2][1]
y=np.array([[3,4],[0,2],[3,0]])
print(y)
print(f2(y))
print(numerical_grad(f2,y))