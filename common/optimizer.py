"最適化手法のクラス群"

class SGD:
    "Stochastic Gradient Descent"
    def __init__(self, lr=0.01):
        self.lr = lr # 学習係数η. Learning Rate
    def update(self, params, grads):
        for prm, grd in zip(params, grads):
            prm -= self.lr * grd