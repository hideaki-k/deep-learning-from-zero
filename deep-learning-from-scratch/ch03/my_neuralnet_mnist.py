# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) 
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network =pickle.load(f)
    return network

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

"""
def init_network():
    network = {}
    network["w1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["b1"] = np.array([0.1,0.2,0.3])
    network["w2"] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network["b2"] = np.array([0.1,0.2])
    network["w3"] = np.array([[0.1,0.3],[0.2,0.4]])
    network["b3"] = np.array([0.1,0.2])

    return network
"""

def predict(network, x):

    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    z3 = sigmoid(a3)
    y = softmax(a3)

    return y
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
print("Accuracy : " + str(float(accuracy_cnt) / len(x)))