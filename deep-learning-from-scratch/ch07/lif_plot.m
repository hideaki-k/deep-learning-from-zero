t_rc = 0.02
tc_m = 1 % �����萔 (s);
R = 1 %����R;
vthr = -50 % 臒l�d�� (mV);
vrest = -65 % ���Z�b�g�d��(mV);
ganma = 0.02 % �X���[�W���O�W��;
tref = 0.001 % �s���� (s);
I_max = 600 % �ő�d��(nA);
I_min = -600 ;
I = I_min:0.01:I_max;
%rate = 1 / (tref + tc_m*log(1 + ((vthr - vrest)/(ganma*log(1 + ((I*R - vthr + vrest)/ganma)))))) + I;
%rate = (tref + tc_m*log(1 + ((vthr - vrest)/(ganma*log(1 + exp((I*R - vthr + vrest)/ganma))))))
rate = 1 / (tref + t_rc*log(1 + ((vthr-vrest) / (ganma*log(1 + (exp((I*R - vthr + vrest)/ganma)))))))
%rate = sin(I)
plot(I,rate)