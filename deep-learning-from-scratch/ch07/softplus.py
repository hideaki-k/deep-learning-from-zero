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

k = 0.19
a = 0.2 # ???????
b = 1/(k*a)  # k???????
sfactor = 1. # 208.76 #49.66 ?????????sfactor=1???
I_max = 0.5
x = np.arange(-0.5, I_max, 0.01) #入力電浝

y_2 = k * a * b * x * sfactor
y_5 = k * a * b * x * sfactor
y_8 = k * a * b * x * sfactor
y_10 = k * a * b * x * sfactor
y_15 = k * a * b * x * sfactor
y_20 = k * a * b * x * sfactor 

y_2[x<10] = sfactor * k * a * np.log(1.+ np.exp(x[x<10]*b))
y_5[x<10] = sfactor * k * 0.5 * np.log(1.+ np.exp(x[x<10]/(k*0.5)))
y_8[x<10] = sfactor * k * 0.8 * np.log(1.+ np.exp(x[x<10]/(k*0.8)))
y_10[x<10] = sfactor * k * 1 * np.log(1.+ np.exp(x[x<10]/(k*1)))
y_15[x<10] = sfactor * k * 1.5 * np.log(1.+ np.exp(x[x<10]/(k*1.5)))
y_20[x<10] = sfactor * k * 2 * np.log(1.+ np.exp(x[x<10]/(k*2)))

#y = y - np.log(2)
#y = 14.*np.log(1.+np.exp(output*10.))
fig = plt.figure(figsize=(4, 3))
plt.plot(x, y_2, color="k",label='0.2')
plt.plot(x, y_5, color="k",linestyle = "--" ,label='0.5')
plt.plot(x, y_8, color="k",linestyle = "dashed",label='0.8')
plt.plot(x, y_10, color="k",linestyle = "dashdot",label='1.0')
plt.plot(x, y_15, color="k", linestyle = "dotted",label='1.5')
#plt.plot(x, y_20, color="k",linestyle = "solid")

ax = fig.add_subplot(111)
# x??????????
ax.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 0.5)

# y????????
ax.grid(which = "major", axis = "y", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 0.5)
ax.legend()
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate (Hz)')
plt.xlim(-0.5, I_max)
plt.show()
"""
a = 0.038
b = 26.315
sfactor = 1#49.66
I_max = 0.5
x = np.arange(-0.5, I_max, 0.01) #????

y = a * b * x * sfactor;
y[x<10] = sfactor * a * np.log(1.+ np.exp(x[x<10]*b))

#y = y - np.log(2)
#y = 14.*np.log(1.+np.exp(output*10.))
plt.figure(figsize=(4, 3))
plt.plot(x, y, color="k")
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate (Hz)')
plt.xlim(-0.5, I_max)
plt.show()
"""