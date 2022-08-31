import numpy as np


class Quaternion:

    @staticmethod
    def norm(v):
        return np.sqrt(np.sum(np.square(v)))

    @staticmethod
    def quat_prod(q, r):
        t = np.zeros(4)
        t[0] = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]
        t[1] = q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]
        t[2] = q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1]
        t[3] = q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]
        if Quaternion.norm(t) == 0:
            return np.array([0, 0, 0, 0])
        return t / Quaternion.norm(t)

    @staticmethod
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_log(q):
        if Quaternion.norm(q[1:]) == 0:
            return np.array([0, 0, 0])
        return 2 * np.arccos(q[0]) / Quaternion.norm(q[1:]) * q[1:]

    @staticmethod
    def quat_exp(v):
        t = np.sin(0.5 * Quaternion.norm(v)) / Quaternion.norm(v) * v
        return np.array([np.cos(0.5 * Quaternion.norm(v)), t[0], t[1], t[2]])

    @staticmethod
    def quat_dist(q, r):
        return 2 * Quaternion.quat_log(Quaternion.quat_prod(q, Quaternion.quat_conj(r)))
