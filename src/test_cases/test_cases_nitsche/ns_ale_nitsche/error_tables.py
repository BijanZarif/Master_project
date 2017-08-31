import matplotlib
matplotlib.use('webagg')
matplotlib.rc('webagg', port = 8000, open_in_browser = False)

from numpy import *
from math import *
#from matplotlib import pyplot as plt
from fenics import *
from mshr import *   # I need this if I want to use the functions below to create a mesh

N = [2**3, 2**4, 2**5]
DT = [1./(float(N[i])) for i in range(len(N))]
#DT2 = [1./(float(N[i])**2) for i in range(len(N))]

rates_u0 = []
rates_p0 = []
rates_u = []
rates_w = []
rates_p = []

rates_u_0N = []
rates_p_0N = []

rates_u_N = []
rates_w_N = []
rates_p_N = []

u_err_H1_0 = [2.0108e-01, 5.1827e-02, 1.2948e-02]
p_err_L2_0 = [2.2342e-03, 4.2361e-04, 1.0166e-04]

u_err_H1 = [3.1954e-01, 2.0958e-01, 2.0205e-01]
w_err_H1 = [2.5965e-02, 1.3038e-02, 6.5604e-03]
p_err_L2 = [1.6220e-02, 1.7039e-02, 1.7759e-02]

u_err_H1_0N = [2.0115e-01, 5.1839e-02, 1.2949e-02]
p_err_L2_0N = [2.5986e-03, 4.4706e-04, 1.0312e-04]

u_err_H1_N = [ 3.2014e-01, 2.0962e-01, 2.0206e-01]
w_err_H1_N = [ 2.5965e-02, 1.3038e-02, 6.5604e-03]
p_err_L2_N = [ 1.6777e-02, 1.7059e-02, 1.7761e-02]

for i in range(len(DT)-1):
    rates_u0.append(math.log(u_err_H1_0[i+1]/u_err_H1_0[i])/math.log(DT[i+1]/DT[i]) )
    rates_p0.append(math.log(p_err_L2_0[i+1]/p_err_L2_0[i])/math.log(DT[i+1]/DT[i]) )
    rates_u.append(math.log(u_err_H1[i+1]/u_err_H1[i])/math.log(DT[i+1]/DT[i]) )
    rates_w.append(math.log(w_err_H1[i+1]/w_err_H1[i])/math.log(DT[i+1]/DT[i]) )
    rates_p.append(math.log(p_err_L2[i+1]/p_err_L2[i])/math.log(DT[i+1]/DT[i]) )
    rates_u_0N.append(math.log(u_err_H1_0N[i+1]/u_err_H1_0N[i])/math.log(DT[i+1]/DT[i]) )
    rates_p_0N.append(math.log(p_err_L2_0N[i+1]/p_err_L2_0N[i])/math.log(DT[i+1]/DT[i]) )
    rates_u_N.append(math.log(u_err_H1_N[i+1]/u_err_H1_N[i])/math.log(DT[i+1]/DT[i]) )
    rates_w_N.append(math.log(w_err_H1_N[i+1]/w_err_H1_N[i])/math.log(DT[i+1]/DT[i]) )
    rates_p_N.append(math.log(p_err_L2_N[i+1]/p_err_L2_N[i])/math.log(DT[i+1]/DT[i]) )    
print rates_u0, rates_p0, rates_u, rates_w, rates_p, rates_u_0N, rates_p_0N, rates_u_N, rates_w_N, rates_p_N


# Errors w = 0 without Nitsche
u_errors =  [['2.0108e-01', '5.3410e-02', '1.3308e-02'],
             ['1.9491e-01', '5.1827e-02', '1.3130e-02'],
             ['1.9423e-01', '5.0625e-02', '1.2948e-02']]

w_errors =  [['0.0000e+00', '0.0000e+00', '0.0000e+00'],
             ['0.0000e+00', '0.0000e+00', '0.0000e+00'],
             ['0.0000e+00', '0.0000e+00', '0.0000e+00']]

p_errors =  [['2.2342e-03', '1.5843e-03', '1.5946e-03'],
             ['1.6935e-03', '4.2361e-04', '3.9789e-04'],
             ['1.6297e-03', '1.9945e-04', '1.0166e-04']]

# Errors w != 0 without Nitsche
u_errors =  [['3.1954e-01', '2.1620e-01', '2.0647e-01'],
             ['3.1298e-01', '2.0958e-01', '1.9979e-01'],
             ['3.1478e-01', '2.1150e-01', '2.0205e-01']]

w_errors =  [['2.5965e-02', '2.5389e-02', '2.5350e-02'],
             ['1.4080e-02', '1.3038e-02', '1.2968e-02'],
             ['8.5368e-03', '6.6945e-03', '6.5604e-03']]

p_errors =  [['1.6220e-02', '1.5294e-02', '1.5273e-02'],
             ['1.7914e-02', '1.7039e-02', '1.7013e-02'],
             ['1.8622e-02', '1.7785e-02', '1.7759e-02']]

# Errors w = 0 with Nitsche
u_errors =  [['2.0115e-01', '5.3419e-02', '1.3313e-02'],
             ['1.9498e-01', '5.1839e-02', '1.3131e-02'],
             ['1.9430e-01', '5.0641e-02', '1.2949e-02']]

w_errors =  [['0.0000e+00', '0.0000e+00', '0.0000e+00'],
             ['0.0000e+00', '0.0000e+00', '0.0000e+00'],
             ['0.0000e+00', '0.0000e+00', '0.0000e+00']]

p_errors =  [['2.5986e-03', '1.5928e-03', '1.5949e-03'],
             ['2.1029e-03', '4.4706e-04', '3.9847e-04'],
             ['2.0305e-03', '2.4174e-04', '1.0312e-04']]

# Errors w != 0 with Nitsche
u_errors =  [['3.2014e-01', '2.1621e-01', '2.0648e-01'],
             ['3.1388e-01', '2.0962e-01', '1.9980e-01'],
             ['3.1581e-01', '2.1155e-01', '2.0206e-01']]

w_errors =  [['2.5965e-02', '2.5389e-02', '2.5350e-02'],
             ['1.4080e-02', '1.3038e-02', '1.2968e-02'],
             ['8.5368e-03', '6.6945e-03', '6.5604e-03']]

p_errors =  [['1.6777e-02', '1.5310e-02', '1.5274e-02'],
             ['1.8533e-02', '1.7059e-02', '1.7014e-02'],
             ['1.9263e-02', '1.7807e-02', '1.7761e-02']]
