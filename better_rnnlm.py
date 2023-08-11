# Rnnlm改良
# LSTMレイヤを2層利用し、各層にDropoutを使うモデル
import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *  # import numpy as np
import pickle

class BetterRnnlm:
    def __init__(self, vocab_size=10000, wordvec_size=100,hidden_size=100, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn # 標準正規分布に従う乱数

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        #
        # self.layers = [
        #     TimeEmbedding(embed_W),
        #     TimeDropout(dropout_ratio),
        #     TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
        #     TimeDropout(dropout_ratio),
        #     TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
        #     TimeDropout(dropout_ratio),
        #     TimeAffine(embed_W.T, affine_b)  # weight tying!!
        # ]
        self.layers = [
            TimeEmbedding(embed_W),
            VariationalDropout(dropout_ratio),  # 変分Dropoutに変更
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            VariationalDropout(dropout_ratio),  # 変分Dropoutに変更
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            VariationalDropout(dropout_ratio),  # 変分Dropoutに変更
            TimeAffine(embed_W.T, affine_b)
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return  dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

    def save_params(self, file_name="./models/BetterRnnlm.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="./models/BettrRnnlm.pkl"):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)