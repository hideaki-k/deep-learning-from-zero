# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from my_two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
#教師データ
#x_train　 =60000*784
#t_train   =60000*10

#x_train[0].shape=60000

#ハイパーパラメータ
iters_num = 1
train_size = x_train.shape[0]#60000
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


batch_mask = np.random.choice(train_size, batch_size)#0~60000中の乱数,1*100行列生成
print(f"batch_mask.shape:{batch_mask.shape}")
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(f"x_batch.size{x_batch.size}")
print(f"t_batch.size{t_batch.size}")

for i in range(iters_num):
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    #x_batch = 100*784
    #t_batch = 100*10

    #勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    

    #パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    #学習過程の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)