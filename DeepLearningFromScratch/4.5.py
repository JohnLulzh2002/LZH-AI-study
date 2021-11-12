import numpy as np
from matplotlib import pyplot as plt
from Dataset.mnist_mod import load_mnist
from CommonFun import sigmoid, sigmoid_grad, softmax, cross_entropy_error, numerical_gradient
# a=np.array([0.2,0.8,0])
# b=np.array([0.3,0.4,0.3])
# c=np.array([0,1,0])
# t=np.array([0,1,0])
# print(cross_entropy_error(a,t))
# print(cross_entropy_error(b,t))
# print(cross_entropy_error(c,t))
# exit()


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_num = x.shape[0]
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
iters_times = 1000  #10000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.1
network = TwoLayerNet(784, 50, 10)
train_loss_list = []
acc_list = []

for i in range(iters_times):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % 5 == 0:
        print(i, ':', loss, end='    ')
        test_size = x_test.shape[0]
        batch_mask = np.random.choice(test_size, batch_size)
        x_batch = x_test[batch_mask]
        t_batch = t_test[batch_mask]
        print(network.accuracy(x_batch, t_batch) * 100)

    test_size = x_test.shape[0]
    batch_mask = np.random.choice(test_size, batch_size)
    x_batch = x_test[batch_mask]
    t_batch = t_test[batch_mask]
    acc_list.append(network.accuracy(x_batch, t_batch) * 100)

xaxis = np.array(range(len(acc_list)))
plt.plot(xaxis, acc_list, label='acc')
xaxis = np.array(range(len(train_loss_list)))
plt.plot(xaxis, train_loss_list, label='loss')
plt.legend()
plt.show()