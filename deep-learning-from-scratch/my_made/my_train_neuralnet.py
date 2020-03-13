# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
print("yes")