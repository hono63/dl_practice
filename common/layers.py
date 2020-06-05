"Library of Layers"

import copy
import numpy as np

class Sigmoid:
    "Sigmoid Layer. y = 1 / (1 + exp(-x))"
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None # 順伝播の出力
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out) # dy/dx = y(1-y).  xへの逆伝播は dL/dy*dy/dx. ここでdL/dyがdout.
        return dx


class Affine:
    "全結合層 y = xW + b"
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads  = [np.zeros_like(W), np.zeros_like(b)]
        self.x      = None
    def forward(self, x):
        self.x = x
        W, b   = self.params
        return np.dot(x, W) + b
    def backward(self, dout):
        W, b   = self.params
        dx = np.dot(dout, W.T) # dy/dL * Wの転置
        dW = np.dot(dout, self.x.T) # dy/dL * xの転置
        db = np.sum(dout, axis=0) # dy/dLの全行を足し合わせて1行に
        self.grads[0] = copy.deepcopy(dW)
        self.grads[1] = copy.deepcopy(db)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.t = None # Teacher Data
        self.y = None # Output Data
    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1) # argmaxは最大値のあるindexを返す
        loss = cross_entropy(self.y, self.t)
        return loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx

def softmax(x):
    "xは1次元のndarrayベクトル"
    return np.exp(x) / np.sum(x)

def cross_entropy(y, t):
    "交差エントロピー誤差. yは出力値、tは教師データ"
    return -np.sum(t * np.log(y))
