"""
rng.py

A little module for Random Number Generation from a given probability density function
using inverse transform sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from pynverse import inversefunc
import warnings
from tqdm import tqdm

def pdf(x):
    """user-input probability density function"""
    return 2 * np.exp(-2 * x )

def integral(f, a, b):
    """general function to give integral of f from a to b"""
    return integrate.quad(lambda x: f(x), a, b)[0]

def cdf(pdf):
    """given pdf(x), return cdf(x) = int_-infty^x pdf(x')dx'"""
    def eval_cdf(x):
        # function to evaluate cdf
        if hasattr(x, "__iter__"):  # for iterables so you can call with arrays
            return np.array([integral(pdf, -np.inf, xi) for xi in x])
        return np.array(integral(pdf, -np.inf, x))  # if called with single values
    return eval_cdf

def sample(cdf, n):
    # get n points from uniform [0,1] dist
    r = np.random.uniform(0,1,n) 
    # get inverse of cdf
    cdf_inv = inversefunc(cdf)
    # error messages? pfft, who needs em?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # generate random points
        generated = np.empty(n)
        for i in tqdm(range(n)):
            generated[i] = cdf_inv(r[i])
    return generated

# make some x values for plotting
x = np.linspace(0, 5, 100)

# make pdf
f = pdf
plt.plot(x,f(x), label='pdf')

# then convert to cdf which definitely has an inverse
F = cdf(pdf)
plt.plot(x,F(x), label='cdf')

# then generate samples from cdf = F ==> pdf = f
print('generating')
generated = sample(F, 10000)
print('plotting')
plt.hist(generated, density=True, label='generated', bins=100)

plt.legend(loc='best')
plt.show()






