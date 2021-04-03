import numpy as np
import os
from CommonFun import cross_entropy_error
from Dataset.mnist_mod import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(one_hot_label=True)

batch_mask=np.random.choice(x_train.shape[0],10)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]
print(x_train.shape,x_batch.shape,sep='\t->')
print(t_train.shape,t_batch.shape,sep='\t->')