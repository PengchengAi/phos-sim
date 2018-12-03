# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 08:36:32 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from numpy.linalg import inv
from scipy import optimize

# enable mathematical printing
sp.init_printing()

# initial symbols
t, K, t_0, tau_p, base = sp.symbols("t, K, t_0, tau_p, base", real=True) 

# the function
f = K * (((t - t_0)/tau_p) ** 2) * sp.exp(-2.0 * (t - t_0) / tau_p) + base

# calculate paritial derivates
f_diff_K = f.diff(K)
f_diff_t_0 = f.diff(t_0)
f_diff_tau_p = f.diff(tau_p)
f_diff_base = f.diff(base)

# convert to lambda expressions
fun = sp.lambdify((t, K, t_0, tau_p, base), f)
fun_diff_K = sp.lambdify((t, K, t_0, tau_p, base), f_diff_K)
fun_diff_t_0 = sp.lambdify((t, K, t_0, tau_p, base), f_diff_t_0)
fun_diff_tau_p = sp.lambdify((t, K, t_0, tau_p, base), f_diff_tau_p)
fun_diff_base = sp.lambdify((t, K, t_0, tau_p, base), f_diff_base)

# parameters
start = 0
end = start + 32 * 0.1
value_K = 5.12
value_t_0 = 0.0
value_tau_p = 2.0
value_base = 0.1

# compare one-order approximation and curve fit

# generate variable
x = np.linspace(start, end, 33, endpoint=True)

# matrix J
data_diff_K = fun_diff_K(x, value_K, value_t_0, value_tau_p, value_base)
data_diff_t_0 = fun_diff_t_0(x, value_K, value_t_0, value_tau_p, value_base)
J = np.stack((data_diff_K, data_diff_t_0), axis=-1)

# matrix P
data_diff_tau_p = fun_diff_tau_p(x, value_K, value_t_0, value_tau_p, value_base)
data_diff_base = [fun_diff_base(v, value_K, value_t_0, value_tau_p, value_base)
                  for v in x]
P = np.stack((data_diff_tau_p, data_diff_base), axis=-1)

# linear approximation
JP_pre = inv(J.T @ J) @ J.T @ P

min_delta_base = -0.3
max_delta_base = 0.3
span_delta_base = max_delta_base - min_delta_base
cnt = 30
# base varies
dt_list = []
fit_list = []
cal_list = []
for i in range(cnt + 1):
    delta_theta = np.array((0.0, min_delta_base + span_delta_base / cnt * i)).reshape((2, 1))
    total_tau_p = value_tau_p + delta_theta[0, 0]
    total_base = value_base + delta_theta[1, 0]
    
    # generate fitting data
    data = fun(x, value_K, value_t_0, total_tau_p, total_base)

    # fit results
    fit_fun = lambda t, m, n : fun(t, m, n, value_tau_p, value_base)
    popt, pcov = optimize.curve_fit(fit_fun, x, data, p0=[4.0, 0.1])
    fit_delta_beta = np.array(popt) - np.array((value_K, value_t_0))
    
    # calculation results
    cal_delta_beta = JP_pre @ delta_theta
    
    # storage
    dt_list.append(delta_theta)
    fit_list.append(fit_delta_beta)
    cal_list.append(cal_delta_beta)

# plot
delta_base = [v[1, 0] for v in dt_list]
fit_delta_K = [v[0] for v in fit_list]
cal_delta_K = [v[0, 0] for v in cal_list]
plt.xlabel("shift of base")
plt.ylabel("shift of K in curve fitting")
plt.plot(delta_base, fit_delta_K, "o", label="actual point")
plt.plot(delta_base, cal_delta_K, "-", label="linear approximation")
plt.axvline(x=0, color="grey")
plt.axhline(y=0, color="grey")

plt.legend()
plt.show()

delta_base = [v[1, 0] for v in dt_list]
fit_delta_t_0 = [v[1] for v in fit_list]
cal_delta_t_0 = [v[1, 0] for v in cal_list]
plt.xlabel("shift of base")
plt.ylabel("shift of t_0 in curve fitting")
plt.plot(delta_base, fit_delta_t_0, "o", label="actual point")
plt.plot(delta_base, cal_delta_t_0, "-", label="linear approximation")
plt.axvline(x=0, color="grey")
plt.axhline(y=0, color="grey")

plt.legend()
plt.show()
