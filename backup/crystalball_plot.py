import numpy as np
from scipy.stats import crystalball
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

beta, m = 2, 3
x = np.linspace(crystalball.ppf(0.01, beta, m), crystalball.ppf(0.99, beta, m), 100)
ax.plot(x, crystalball.pdf(x, beta, m), 'r-', alpha=0.6, label='crystalball pdf')
plt.show()