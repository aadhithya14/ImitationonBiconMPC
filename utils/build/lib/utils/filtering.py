import numpy as np


def remove_outliers(x, est_len, max_std):
	for i in range(est_len, x.shape[0] - 1):
		max_diff = max_std * np.std(x[i - est_len : i - 1])
		if x[i] < x[i - 1] - max_diff or x[i] > x[i - 1] + max_diff:
			x[i] = x[i - 1]
	return x

def smooth(x, width):
	y = x.copy()
	for i in range(width, x.shape[0]):
		y[i] = np.mean(x[i - width : i - 1])
	return y