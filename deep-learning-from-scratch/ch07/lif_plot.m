tc_m = 1 % 膜時定数 (s);
R = 1 %膜抵抗;
vthr = -50 % 閾値電位 (mV);
vrest = -65 % リセット電圧(mV);
ganma = 0.02 % スムージング係数;
tref = 0.001 % 不応期 (s);
I_max = 600 % 最大電流(nA);
I_min = -600 ;
I = I_min:0.01:I_max;
%rate = 1 / (tref + tc_m*log(1 + ((vthr - vrest)/(ganma*log(1 + ((I*R - vthr + vrest)/ganma)))))) + I;
%rate = (tref + tc_m*log(1 + ((vthr - vrest)/(ganma*log(1 + exp((I*R - vthr + vrest)/ganma))))))
rate = 1 / (tref + tc_m*log(1 + (vthr - vrest) / ganma*log(1 + exp((I*R - vthr + vrest)/ganma))))
%rate = sin(I)
plot(I,rate)