from common.np import *
from common.config import GPU

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # Ellipsis : 次元省略
        if GPU:
            np.scatter_add(dW, self.idx, dout) # dWのidxにdoutを加算
        else:
            np.add.at(dW, self.idx, dout)
        return None