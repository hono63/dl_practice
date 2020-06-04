
import copy
import numpy as np
import matplotlib.pyplot as plt

from dataset import spiral 

class Sigmoid:
    "Sigmoid Layer. y = 1 / (1 + exp(-x))"
    def __init__(self):
        self.params = []
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
        self.params, self.grad = [], []
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


class TwoLayerNet:
    "Affineを二層つなげたネットワーク"
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # Initialize Weight & Bias
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        # Generate Layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        # Aggregate all layers' parameters
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

def softmax(x):
    "xは1次元のndarrayベクトル"
    return np.exp(x) / np.sum(x)

def cross_entropy(y, t):
    "交差エントロピー誤差. yは出力値、tは教師データ"
    return -np.sum(t * np.log(y))

def show_sigmoid():
    x = np.arange(-5, 5, 0.1)
    y = 1.0 / (1.0 + np.exp(-x)) # これでyもndarrayになる
    plt.plot(x,y)
    plt.show()

def show_spiral():
    x, t = spiral.load_data()
    #print(x.shape, t.shape)
    plt.scatter(x[0], x[1])
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
    show_spiral()
    #all_connection()