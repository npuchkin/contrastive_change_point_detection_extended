# CUSUM algorithm for sequential change point detection

import numpy as np


def compute_cusum(X):
    T = np.zeros(X.shape[0])
    for n in range(1, X.shape[0]):
        t = np.zeros(n)
        for k in range(1, n):

            t[k] = abs((n - k) * X[:k].sum()   - k * X[k:n].sum() ) / np.sqrt(n * k * (n - k))
          # t = abs(np.mean(X[:k]) - np.mean(X[k:n]))
        T[n] = np.max(t)
    return T
    
    
def compute_cusum_squared(X):
    T = np.zeros(X.shape[0])
    for n in range(1, X.shape[0]):
        t = np.zeros(n)
        for k in range(1, n):

            t[k] = abs((n - k) * np.power(X[:k], 2).sum()   - k * np.power(X[k:n], 2).sum() ) / np.sqrt(n * k * (n - k))
          # t = abs(np.mean(X[:k]) - np.mean(X[k:n]))
        T[n] = np.max(t)
    return T