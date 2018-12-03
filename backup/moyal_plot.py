import numpy as np
from scipy.stats import moyal, crystalball
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

# plot moyal pdf
loc, scale = 0, 0.625
x = np.linspace(moyal.ppf(0.01, loc, scale), moyal.ppf(0.99, loc, scale), 100)
ax.plot(x, moyal.pdf(x, loc, scale), 'r-', alpha=0.6, label='moyal pdf')

# plot crystal ball pdf
beta, m = 2, 3
x = np.linspace(crystalball.ppf(0.01, beta, m), crystalball.ppf(0.99, beta, m), 100)
ax.plot(x, crystalball.pdf(x, beta, m), 'b-', alpha=0.6, label='crystalball pdf')

plt.legend()
plt.show()
