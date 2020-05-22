# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_rc = 0.02
tc_m = 20 # 膜時定数 (s)
R = 1 #膜抵抗
vthr = 1 # 閾値電位 (mV)
vrest = -65 # リセット電圧(mV)
ganma = 0.02 # スムージング係数
tref = 0.004 # 不応期 (s)
I_max = 8 # 最大電流(nA)
I_min = 0

I = np.arange(I_min, I_max, 0.01) #入力電流
rate = 1 / (tref + t_rc*np.log(1 + (vthr / (ganma*np.log(1 + np.exp((I - vthr)/ganma))))))
#rate[np.isnan(rate)] = 0 # nan to 0
# 描画
plt.figure(figsize=(4, 3))
plt.plot(I, rate, color="k")
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate (Hz)')
plt.xlim(I_min, I_max)
plt.show()
