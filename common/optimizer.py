"最適化手法のクラス群"

class SGD:
    "Stochastic Gradient Descent"
    def __init__(self, lr=0.01):
        self.lr = lr # 学習係数η. Learning Rate
    def update(self, params_list, grads_list):
        "params shall be updated by grads"
        for params, grads in zip(params_list, grads_list):
            for key in params.keys():
                params[key] = params[key] - self.lr * grads[key]