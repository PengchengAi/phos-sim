import numpy as np
from scipy.stats import moyal, norm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

# plot moyal pdf
loc, scale = 0.004, 0.006
x = np.linspace(moyal.ppf(0.01, loc=loc, scale=scale), moyal.ppf(0.99, loc=loc, scale=scale), 100)
ax.plot(x, moyal.pdf(x, loc=loc, scale=scale), 'r', alpha=0.6, label='moyal pdf')

# plot moyal samples
s_moyal = moyal.rvs(loc, scale, size=1000)
ax.hist(s_moyal, 40, alpha=0.6, density=True, label='moyal histogram')

# plot normal pdf
mu, sigma = 0, 0.014 # mean and standard deviation
x = np.linspace(norm.ppf(0.01, loc=mu, scale=sigma), norm.ppf(0.99, loc=mu, scale=sigma), 100)
ax.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'b', alpha=0.6, label='normal pdf')

# plot normal samples
s_normal = np.random.normal(mu, sigma, 1000)
ax.hist(s_normal, 40, alpha=0.6, density=True, label='normal histogram')

ax.axvline(x=0)

plt.legend()
plt.show()
