import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pickle
sys.path.append("Dataset")
from mnist_mod import load_mnist
from CommonFun import forward,sigmoid,softmax

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False)

print('x_train.shape=\t',x_train.shape)
print('t_train.shape=\t',t_train.shape)
print('x_test.shape=\t',x_test.shape)
print('t_test.shape=\t',t_test.shape)
print()

def show(img):
    Image.fromarray(np.uint8(img)).show()
img,label=x_train[0],t_train[0]
print(label)
print(img.shape, end='->')
img=img.reshape(28,28)
print(img.shape)
plt.imshow(Image.fromarray(np.uint8(img)), cmap='gray')
plt.show()

with open(os.path.join('Dataset',"sample_weight.pkl"), 'rb') as f:
    network = pickle.load(f)
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    z1 = forward(x,W1,b1)
    z2 = forward(z1,W2,b2)
    z3 = forward(z2,W3,b3)
    return z3

accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p= np.argmax(y)
    if p == t_test[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))

x,t= load_mnist(normalize=True)[1]
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    y = predict(network, x[i:i + batch_size])
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))