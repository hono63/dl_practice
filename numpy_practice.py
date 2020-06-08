"Main"

import numpy as np
import matplotlib.pyplot as plt

from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizer import SGD
from dataset import spiral 

class TwoLayerNet:
    "Affineを二層つなげたネットワーク"
    def __init__(self, input_size, hidden_size, output_size):
        """
        class initializer

        Parameters
        --------
        input_size : int
            the number of input neurons
        hidden_size : int
            the number of hidden neurons
        output_size : int
            the number of output neurons
        """
        I, H, O = input_size, hidden_size, output_size
        # Initialize Weight & Bias
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        # Generate Layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()
        # Store all layers' parameters (オリジナルとは異なる)
        self.params_list, self.grads_list = [], []
        for layer in self.layers:
            self.params_list.append(layer.params)
            self.grads_list.append(layer.grads)
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def forward(self, x, t):
        """
        Parameters
        --------
        x: ndarray
            input of the highest layer
        t: ndarray
            teacher data

        Returns
        --------
        loss: ndarray
            loss of prediction result
        """
        score = self.predict(x)
        loss  = self.loss_layer.forward(score, t)
        return loss
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

def train_custom_loop():
    # Hyper parameters
    MAX_EPOCH = 300
    BATCH_SIZE = 30
    HIDDEN_SIZE = 10
    LEARNING_RATE = 1.0
    # Load data and generate optimizer
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=HIDDEN_SIZE, output_size=3)
    optimizer = SGD(lr=LEARNING_RATE)
    # variables used in learning
    data_size = len(x)
    max_iters = data_size // BATCH_SIZE
    total_loss = 0
    loss_count = 0
    loss_list = []
    # data shuffle
    for epoch in range(MAX_EPOCH):
        idx = np.random.permutation(data_size) # 0~data_sizeの数字をランダムに並べ替えたndarrayを生成する
        x = x[idx]
        t = t[idx]
        for iters in range(max_iters):
            batch_x = x[iters*BATCH_SIZE : (iters+1)*BATCH_SIZE]
            batch_t = t[iters*BATCH_SIZE : (iters+1)*BATCH_SIZE]
            # update gradients and parameters
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params_list, model.grads_list)
            total_loss += loss
            loss_count += 1
            # print learning progress
            if (iters+1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print('| epoch %d | iter %d / %d | loss %.2f |' % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0

def show_sigmoid():
    x = np.arange(-5, 5, 0.1)
    y = 1.0 / (1.0 + np.exp(-x)) # これでyもndarrayになる
    plt.plot(x,y)
    plt.show()

def show_spiral():
    x, t = spiral.load_data()
    #print(x.shape, t.shape)
    plt.scatter(x[0:100, 0], x[0:100, 1])
    plt.scatter(x[100:200, 0], x[100:200, 1])
    plt.scatter(x[200:300, 0], x[200:300, 1])
    plt.show()

def 基本():
    x = np.array([1,2,3])
    print(x.__class__)
    print(x.shape)
    print(x.ndim)

    W = np.array([[1,2,3], [4,5,6]])
    print(W.__class__)
    print(W.shape)
    print(W.ndim)

    X = np.array([[0,1,2], [3,4,5]])
    print(X+W)
    print(X*W) # 要素ごとの掛け算になる
    print(W*np.array([10,20,30])) # 勝手に拡張されてW * [[10,20,30],[10,20,30]] の掛け算になる ちょっとキモチワルイ
    # ベクトルの内積
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    print(np.dot(a, b))
    # 行列の積
    A = np.array([[1,2],[3,4]])
    B = np.array([[5,6],[7,8]])
    print(np.dot(A, B))

def all_connection():
    W1 = np.random.randn(2,4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4,3)
    b2 = np.random.randn(3)
    x  = np.random.randn(10,2) # 10個のサンプルデータ
    h  = np.dot(x,W1) + b1
    #print(W1)
    #print(b1)
    #print(x)
    print(h)
    a  = Sigmoid.forward(h)
    s  = np.dot(a,W2) + b2
    print(s)

if __name__=="__main__":
    #show_sigmoid()
    #show_spiral()
    #all_connection()
    train_custom_loop()