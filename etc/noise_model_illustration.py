import matplotlib.pyplot as plt
import numpy as np


def pcm_noise_sigma(x):

    #The numbers here are based on the paper https://arxiv.org/pdf/1906.03138.pdf, Figure 3 a,b

    if x <= .1:
        noise_std = x * .275/.1 + .25
    elif x<=.4:
        noise_std = x * .3/.3 + .5
    elif x <= .8:
        noise_std = (x-.4) * (1.1-.825)/.4 + .825
    else:
        noise_std = 1.125

    return noise_std

n_repetitions = 10000

values = np.array([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

wts = values.flatten()
stds = np.array([pcm_noise_sigma(wt_) for wt_ in wts])

plt.figure()

plt.plot(wts , stds, '-b*' )
plt.grid(True)
plt.title('Weight Value vs Stddev')
plt.xlabel('Weight Value')
plt.ylabel('Standard Dev of Noise')

import math
import scipy.stats as stats

plt.figure()

for val, sig in zip(wts, stds):
    mu = val
    sigma = sig/25
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    print(stats.norm.pdf(x, mu, sigma))
    # exit()
    plt.plot(x, stats.norm.pdf(x, mu, sigma)/(sum(stats.norm.pdf(x, mu, sigma))/max(stats.norm.pdf(x, mu, sigma))))

plt.title('Resulting Weight Distribution for \nExample Set of Weight Values Due to PCM Analog Noise')
plt.grid(True)
plt.ylabel('Density')
plt.xlabel('Example Weight Value')
plt.show()


