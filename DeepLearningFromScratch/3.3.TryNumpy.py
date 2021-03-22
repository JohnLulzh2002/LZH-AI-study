import numpy as np
arr=np.array
a=arr([[1,2],[3,4],[5,6]])
print(a)
print(np.ndim(a))
print(a.shape)
print()
b=arr([[1,2,3],[4,5,6]])
print(np.dot(a,b))
print()
v=arr([2,4,6])
print(np.dot(v,a))