import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np



import scipy.stats as stats

mu1 = 0
mu2 = 3
sigma_x = 1

sigma_y = 3
x1 = np.linspace(mu1 - 3 * sigma_x, mu1 + 3 * sigma_x, 300)
x2 = np.linspace(mu2 - 3 * sigma_x, mu2 + 3 * sigma_x, 300)


y1 = np.linspace(mu1 - 3 * sigma_y, mu1 + 3 * sigma_y, 300)
y2 = np.linspace(mu2 - 3 * sigma_y, mu2 + 3 * sigma_y, 300)
# exit()
plt.subplot(211)
plt.plot(x1, stats.norm.pdf(x1, mu1, sigma_x) / (sum(stats.norm.pdf(x1, mu1, sigma_x)) / max(stats.norm.pdf(x1, mu1, sigma_x))))
plt.plot(x2, stats.norm.pdf(x2, mu2, sigma_x) / (sum(stats.norm.pdf(x2, mu2, sigma_x)) / max(stats.norm.pdf(x2, mu2, sigma_x))))
plt.title('Distributions with Low Overlap: High KL-divergence')
# plt.figure()
# plt.plot(x, stats.norm.pdf(x, mu, sigma) / (sum(stats.norm.pdf(x, mu, sigma)) / max(stats.norm.pdf(x, mu, sigma))))

plt.subplot(212)
plt.plot(y1, stats.norm.pdf(y1, mu1, sigma_y) / (sum(stats.norm.pdf(y1, mu1, sigma_y)) / max(stats.norm.pdf(y1, mu1, sigma_y))))
plt.plot(y2, stats.norm.pdf(y2, mu2, sigma_y) / (sum(stats.norm.pdf(y2, mu2, sigma_y)) / max(stats.norm.pdf(y2, mu2, sigma_y))))
plt.title('Distributions with High Overlap: Low KL-divergence')

plt.show()




