import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from numpy.linalg import inv
from scipy import optimize
from scipy.stats import moyal

import hurdle

# enable mathematical printing
sp.init_printing()

# initial symbols
t, K, t_0, tau_p, base = sp.symbols("t, K, t_0, tau_p, base", real=True) 

# the function
f = K * (((t - t_0)/tau_p) ** 2) * sp.exp(-2.0 * (t - t_0) / tau_p) + base

# calculate paritial derivates
f_diff_K = f.diff(K)
f_diff_t_0 = f.diff(t_0)

# convert to lambda expressions
fun = sp.lambdify((t, K, t_0, tau_p, base), f)
fun_diff_K = sp.lambdify((t, K, t_0, tau_p, base), f_diff_K)
fun_diff_t_0 = sp.lambdify((t, K, t_0, tau_p, base), f_diff_t_0)

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

# linear approximation
J_pre = inv(J.T @ J) @ J.T

"""origin scale: 0.625
   location shift: 2.0 - 4.0   -->   location shift: 3.2 - 6.4
   scale: 0.01                       scale: 0.00625

"""
loc_orig, scale_orig = 0, 0.625
loc = 2.0
scale = 0.01

max_delta_loc = 2.0
cnt = 10

# calcuate mean value using linear approximation
cal_loc_list = []
cal_mean_list = []
cal_std_list = []
for i in range(cnt + 1):
    # use hurdle model to simulate
    total_loc = loc + max_delta_loc / cnt * i
    noise_orig = moyal.rvs(loc_orig, scale_orig, size=3300)
    cal_list = []
    for j in range(100):
        noise = hurdle.hurdle_func_batch(lambda x : (x + total_loc) * scale, noise_orig[j*33:(j+1)*33])
        cal_delta_beta = J_pre @ noise
        cal_list.append(cal_delta_beta)
    cal_list = np.array(cal_list)

    # deduce mean and standard deviation
    cal_mean = np.mean(cal_list, axis=0)
    cal_std = np.std(cal_list, axis=0)
    cal_mean_list.append(cal_mean)
    cal_std_list.append(cal_std)

    cal_loc_list.append((total_loc - loc) * scale)

    print("finish %d cycle(s)" % (i + 1))

samples = 1000

# sample points for curve fitting
fit_loc_list = []
fit_delta_beta_list = []
fit_fun = lambda t, m, n : fun(t, m, n, value_tau_p, value_base)
for i in range(samples):
    random_loc = np.random.random()
    total_loc = loc + random_loc * max_delta_loc
    noise_orig = moyal.rvs(loc_orig, scale_orig, size=33)
    noise = hurdle.hurdle_func_batch(lambda x : (x + total_loc) * scale, noise_orig)
    data = fun(x, value_K, value_t_0, value_tau_p, value_base) + noise
    popt, pcov = optimize.curve_fit(fit_fun, x, data, p0=[4.0, 0.1])
    fit_delta_beta = np.array(popt) - np.array((value_K, value_t_0))

    # storage
    fit_delta_beta_list.append(fit_delta_beta)
    fit_loc_list.append((total_loc - loc) * scale)

    # print information
    if not (i+1) % 100:
        print("finish %d cycles" % (i+1))

# convert python list to numpy array
cal_mean_list = np.array(cal_mean_list)
cal_std_list = np.array(cal_std_list)
fit_delta_beta_list = np.array(fit_delta_beta_list)

# plot K
plt.xlabel("shift of noise p.d.f.")
plt.ylabel("shift of K in curve fitting")
cal_delta_K = cal_mean_list[:, 0]
cal_error_K = cal_std_list[:, 0]
plt.plot(cal_loc_list, cal_delta_K, "b-", label="avg. of linear approx.")
plt.fill_between(cal_loc_list, cal_delta_K-cal_error_K, cal_delta_K+cal_error_K, edgecolor="#6CA6CD", facecolor="#6CA6CD", label="error band")

fit_delta_K = fit_delta_beta_list[:, 0]
plt.scatter(fit_loc_list, fit_delta_K, alpha=0.5, color="r", edgecolors="w", label="Monte Carlo simulation")

plt.legend(loc="upper left")
plt.show()

# plot t_0
plt.xlabel("shift of noise p.d.f.")
plt.ylabel("shift of t_0 in curve fitting")
cal_delta_t_0 = cal_mean_list[:, 1]
cal_error_t_0 = cal_std_list[:, 1]
plt.plot(cal_loc_list, cal_delta_t_0, "b-", label="avg. of linear approx.")
plt.fill_between(cal_loc_list, cal_delta_t_0-cal_error_t_0, cal_delta_t_0+cal_error_t_0, edgecolor="#6CA6CD", facecolor="#6CA6CD", label="error band")

fit_delta_t_0 = fit_delta_beta_list[:, 1]
plt.scatter(fit_loc_list, fit_delta_t_0, alpha=0.5, color="r", edgecolors="w", label="Monte Carlo simulation")

plt.legend(loc="upper right")
plt.show()
