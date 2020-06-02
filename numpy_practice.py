
import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def __init__(self):
        self.params = []
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    def forward(self, x):
        W, b = self.params
        return np.dot(x, W) + b

def show_sigmoid():
    x = np.arange(-5, 5, 0.1)
    y = 1.0 / (1.0 + np.exp(-x)) # これでyもndarrayになる
    plt.plot(x,y)
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
    all_connection()