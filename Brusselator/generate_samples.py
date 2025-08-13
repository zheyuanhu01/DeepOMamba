import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def cov_2exp(t, l, sigma):
    """
    Covariance kernel: squared exponential
    K(i,j) = sigma^2 * exp(- |t_i - t_j|^2 / (2*l^2))

    Parameters:
    t (array-like): Input array of time or input values
    l (float): Length scale parameter
    sigma (float): Standard deviation parameter

    Returns:
    numpy.ndarray: Covariance matrix K
    """
    n = len(t)
    K = np.zeros((n, n))

    # Reshape t to a column vector and create the T1 and T2 matrices
    T1 = np.tile(t.reshape(n, 1), (1, n))
    T2 = np.tile(t.reshape(1, n), (n, 1))

    K = sigma**2 * np.exp(- (T1 - T2)**2 / (2 * l**2))

    return K


def generate_GPsamples(N, sigma, l, t):
    """
    Generate N samples from a Gaussian Process with mean zero and covariance given by cov_2exp.
    
    N: number of samples to generate
    sig: signal variance
    l: length scale
    t: time steps (numpy array)
    """
    Nt = len(t)  # Number of time steps
    mu = np.zeros(Nt)
    K = cov_2exp(t, l, sigma)

    # Draw samples from GP(mu, K)
    S = np.random.multivariate_normal(mu, K, N)

    return S

