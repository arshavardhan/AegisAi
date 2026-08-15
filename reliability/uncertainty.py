import numpy as np

def probability_confidence(p):
    return abs(p - 0.5) * 2

def entropy_uncertainty(p):
    eps = 1e-9
    entropy = -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))
    return entropy / np.log(2)
