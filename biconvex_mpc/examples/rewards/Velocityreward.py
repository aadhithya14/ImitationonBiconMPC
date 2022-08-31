import numpy as np

class Reward():
    def __init__(self,vel,des):
        self.vel= vel
        self.des= des

    def get_reward(self):
        self.reward= 5*np.linalg.norm((self.vel[0]-self.des))
        return self.reward