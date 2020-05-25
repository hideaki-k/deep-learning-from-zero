# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from numpy.random import *
'''
a = 0.2
b = 5.0
y = a * b * x
ind = np.where( (x>-10) & (x<10))[0]
y[ind] = a * np.log( 1 + np.exp(b * x[ind]) )
y[x<=-10] = 0
'''
a = 0.2
b = 5.
sfactor = 1.#49.66
I_max = 0.5
x = np.arange(-0.5, I_max, 0.01) #入力電流

y = a * b * x * sfactor;
y[x<10] = sfactor * a * np.log(1.+ np.exp(x[x<10]*b))

y = y - np.log(2)
#y = 14.*np.log(1.+np.exp(output*10.))
plt.figure(figsize=(4, 3))
plt.plot(x, y, color="k")
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate (Hz)')
plt.xlim(-3, I_max)
plt.show()
