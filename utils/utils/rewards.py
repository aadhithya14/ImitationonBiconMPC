import numpy as np
from utils.my_math import polar_coord, ang_dist

class GoalPositionReward():

    def __init__(self, get_position, k_p, goal_pos, req_dist=None):
        self.get_position = get_position
        self.k_p = k_p
        self.goal_pos = goal_pos
        self.req_dist = req_dist

    def get_reward(self):
        pos = self.get_position()
        return -self.k_p * np.linalg.norm(self.goal_pos - pos)

    def is_done(self):
        if self.req_dist is None:
            return True
        pos = self.get_position()
        return np.linalg.norm(self.goal_pos - pos) < self.req_dist


class RelativePositionReward():

    def __init__(self, get_position_a, get_position_b, k_p, req_dist=None):
        self.get_position_a = get_position_a
        self.get_position_b = get_position_b
        self.k_p = k_p
        self.req_dist = req_dist

    def get_reward(self):
        pos_a = self.get_position_a()
        pos_b = self.get_position_b()
        return [-self.k_p * np.linalg.norm(pos_b - pos_a)]

    def is_done(self):
        if self.req_dist is None:
            return True
        pos_a = self.get_position_a()
        pos_b = self.get_position_b()
        return np.linalg.norm(pos_b - pos_a) < self.req_dist


class PushingIncentive():

    def __init__(self, get_pusher_pos, get_pushee_pos, goal_pos, box_dim, k_i, angle_incentive=False):
        self.get_pusher_pos = get_pusher_pos
        self.get_pushee_pos = get_pushee_pos
        self.goal_pos = goal_pos
        self.box_dim = box_dim
        self.k_i = k_i
        self.angle_incentive = angle_incentive

    def get_reward(self):
        pusher_pos = self.get_pusher_pos()
        pushee_pos = self.get_pushee_pos()
        pusher_r, pusher_theta = polar_coord(pusher_pos - pushee_pos)
        goal_r, goal_theta = polar_coord(self.goal_pos - pushee_pos)

        incentive = -1.0 * pusher_r / (self.box_dim * np.sqrt(2))
        if self.angle_incentive:
            off_angle = ang_dist(pusher_theta, goal_theta + np.pi)
            incentive += -1.0 * off_angle / np.pi

        return self.k_i * incentive

    def is_done(self):
        # As with all incentives, we don't take it into account
        # when checking if the episode is successfully finished
        return True


class DesiredForceIntensityReward():

    def __init__(self, get_pushee_force, k_f, goal_force):
        self.get_pushee_force = get_pushee_force
        self.k_f = k_f
        self.goal_force = goal_force

    def get_reward(self):
        force = self.get_pushee_force()
        # TODO: <eps?
        if force == 0.0:
            return 0.0
        if force < self.goal_force:
            scaling = 2.0
        else:
            scaling = 0.2
        return self.k_f * (0.5 + 0.5 * np.exp(-np.abs((force - self.goal_force) / self.goal_force) * scaling))

    def is_done(self):
        return False


class DesiredForceDirectionReward():

    def __init__(self, get_force, k_d, goal_direction, eps=1e-9):
        self.get_force = get_force
        self.k_d = k_d
        self.goal_direction = goal_direction
        self.eps = eps

    def get_reward(self):
        force = self.get_force()
        if np.linalg.norm(force) < self.eps:
            return 0.0
        force_int, force_angle = polar_coord(force)
        ang_diff = ang_dist(force_angle, self.goal_direction)
        return self.k_d * (np.pi - ang_diff) / np.pi

    def is_done(self):
        return False