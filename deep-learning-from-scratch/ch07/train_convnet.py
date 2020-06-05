# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
import pickle

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 10
"""
# ANN(Relu)の　学習済み重みを読み込み
with open("params.pkl", 'rb') as f:
    trainweight = pickle.load(f)
"""
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
"""
# ANN(Relu)の学習済み重みを読み込み
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, trainweight=trainweight)
"""                      
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=100)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
#x = np.arange(max_epochs)
x = np.arange(len(trainer.train_loss_list))
#print(x)
plt.plot(x, trainer.train_loss_list, label='train', markevery=2)
#plt.plot(x, trainer.test_loss_list, marker='s', label='test', markevery=2)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.0)
plt.legend(loc='lower right')
plt.show()
