"Library of Layers"

import copy
import numpy as np

class Sigmoid:
    "Sigmoid Layer. y = 1 / (1 + exp(-x))"
    def __init__(self):
        self.params, self.grads = {}, {}
        self.out = None # 順伝播の出力
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out) # dy/dx = y(1-y).  xへの逆伝播は dL/dy*dy/dx. ここでdL/dyがdout.
        return dx


class Affine:
    """
    全結合層 y = xW + b
    W : Weight     (Rows: the number of input nodes, Cols: the number of output nodes)
    x : input data (Rows: the number of data, Cols: the number of input nodes)
    b : bias       (Rows: 1, Cols: the number of output nodes)
    y : output     (Rows: the number of data, Cols: the number of output nodes)
    """
    def __init__(self, W, b):
        self.params = {"W": W, "b": b}
        self.grads  = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.x      = None
    def forward(self, x):
        "y = x.W + b"
        self.x = x
        return np.dot(x, self.params["W"]) + self.params["b"]
    def backward(self, dout):
        """
        Paremeters
        --------
        dout : ndarray
            dL/dy matrix (Rows: the number of data, Cols: the number of output nodes)
        --------
        """
        dx = np.dot(dout, self.params["W"].T)      # dL/dy . Wの転置
        dW = np.dot(self.x.T, dout) # x . dL/dyの転置
        db = np.sum(dout, axis=0)   # dL/dyの全行を足し合わせて1行に
        self.grads["W"] = dW.copy()
        self.grads["b"] = db.copy()
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = {}, {}
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
    """
    xは1次元のndarrayベクトル * データ数
    Parameters
    --------
    x : ndarray
    matrix (Rows: the number of data, Cols: dimension of x)
    --------
    """
    expx = np.exp(x)
    return expx / np.sum(expx, axis=1).reshape(30, 1)

def cross_entropy(y, t):
    "交差エントロピー誤差. yは出力値、tは教師データ"
    #return -np.sum(t * np.log(y))
    loss = 0
    for i, t_idx in enumerate(t):
        loss += -np.log(y[i][t_idx])
    return loss
