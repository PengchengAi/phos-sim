import numpy as np

def hurdle_func(func, x):
    if func(x)<=0:
        return 0.0
    else:
        return func(x)

def hurdle_func_batch(func, arr):
    return [hurdle_func(func, v) for v in arr]

def hurdle_mean_variance(func, arr):
    arr_h = hurdle_func_batch(func, arr)
    h_mean = np.mean(arr_h)
    h_var = np.var(arr_h)
    return (h_mean, h_var)
