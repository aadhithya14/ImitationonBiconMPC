import numpy as np
from math import atan2

def polar_coord(v):
    r = np.linalg.norm(v)
    theta = atan2(v[1], v[0])
    return r, theta

def ang_dist(theta1, theta2):
    d = (theta1 - theta2) % (2 * np.pi)
    if d > np.pi:
        d = 2 * np.pi - d
    return d

def scale(x, orig, dest):
    if np.isscalar(x):
        return (x - orig[0]) / (orig[1] - orig[0]) * (dest[1] - dest[0]) + dest[0]
    else:
        x = np.array(x)
        orig = np.array(orig)
        dest = np.array(dest)
        if orig.ndim == 1:
            orig = np.repeat(orig[np.newaxis, :], x.shape[0], axis=0)
        if dest.ndim == 1:
            dest = np.repeat(dest[np.newaxis, :], x.shape[0], axis=0)
        y = np.zeros(x.shape)
        for i in range(x.shape[0]):
            y[i] = (x[i] - orig[i][0]) / (orig[i][1] - orig[i][0]) * (dest[i][1] - dest[i][0]) + dest[i][0]
        return y
