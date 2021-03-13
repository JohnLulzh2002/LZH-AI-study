import numpy as np
arr=np.array
def perceptron(w,x,theta):
    sum=np.dot(w,x.T)-theta
    if sum>0:
        return True
    else:
        return False
def AND(x):
    return perceptron(arr([1,1]),x,1)
def OR(x):
    return perceptron(arr([1,1]),x,0)
def NAND(x):
    return perceptron(arr([-1,-1]),x,-2)
def NOR(x):
    y=arr([OR(x),NAND(x)])
    return AND(y)
a=[arr([0,0]),arr([0,1]),arr([1,0]),arr([1,1])]
for i in a:
    print("AND(",i,")=",AND(i))
print()
for i in a:
    print("OR(",i,")=",OR(i))
print()
for i in a:
    print("NAND(",i,")=",NAND(i))
print()
for i in a:
    print("NOR(",i,")=",NOR(i))