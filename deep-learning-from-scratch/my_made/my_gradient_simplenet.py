# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        #self.W = np.random.randn(2, 3)
        self.W =np.array([[ 0.20639138, 0.45154866,  1.22533566],[-0.37647351,0.03638124,1.00944928]])
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        print(f"z : {z}")
        y = softmax(z)
        print(f"y : {y}")
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w:net.loss(x, t)
dW = numerical_gradient(f,net.W)

#print(f"net.W : {net.W}")
print(f"dW : {dW}")
