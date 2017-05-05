
import numpy as np

def exe(x, noise_mean = 0.0, noise_std = 0.1):
    noise = np.random.randn(*x.shape) * noise_std + noise_mean;
    y = x + noise
    return y, noise
