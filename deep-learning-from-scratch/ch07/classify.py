# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from simple_convnet import SimpleConvNet
from dataset.mnist import load_mnist
from PIL import Image

print(os.pardir)


# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False,normalize=False)

# 処理に時間のかかる場合はデータを削減 
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]
"""
# img表示
img = x_train[0]
label = t_train[0]
print(img.shape)
img = img.reshape(28,28)
pil_img = Image.fromarray(np.uint8(img))
#pil_img.show()
"""
# pretrained weight 読み込み
def init_network():
    with open("params.pkl","rb") as f:
        network = pickle.load(f)

    return network

network = init_network()
#print(network)
# 重みの初期化
params = {}
conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
params["W1"] = network["W1"]
params["b1"] = network["b1"]
params["W2"] = network["W2"]
params["b2"] = network["b2"]
params["W3"] = network["W3"]
params["b3"] = network["b3"]

# レイヤの生成
layers = OrderedDict()
layers["Conv1"] = Convolution(params["W1"], params["b1"],
                                conv_param["stride"], conv_param["pad"])
layers["Relu1"] = Relu()
layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
layers["Affine1"] = Affine(params["W2"], params["b2"])
layers["Relu2"] = Relu()
layers["Affine2"] = Affine(params["W3"], params["b3"])

last_layer = SoftmaxWithLoss()
print(layers.values())

def predict(x):
    for layer in layers.values():
        print(layer)
        x = layer.forward(x)
        #print(x)

    return x

def accuracy(x, t, batch_size=100):
    if t_train.ndim != 1 : t = np.argmax(t, axis=1)

    acc = 0.0
    """
    for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            #print(i)
            #print(tx.shape)
            y = predict(tx)
            
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
    return acc / x.shape[0]
    """
    tx = x[0]
    tt = t[0]
    print(tx.shape)
    tx = tx[np.newaxis,:,:]
    print(tx.shape)
    y = predict(tx)
    print(y)
    y = np.argmax(y, axis=1)
    print(y)



print(x_train.shape)
print(t_train.shape)
test_acc = accuracy(x_train, t_train)
print(test_acc)