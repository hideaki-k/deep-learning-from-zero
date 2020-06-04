# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
fig = plt.figure()

def relu(x):
    return np.maximum(0, x)

ax = fig.add_subplot(111)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
ax.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 0.5)

# y軸に目盛線を設定
ax.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 0.5)
plt.plot(x, y,color='r')
plt.ylim(-2.0, 5.0)
plt.show()
