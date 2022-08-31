from math import isnan
import numpy as np
import math


class SolverTermination():

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        pass

    def step(self):
        pass

    def isNaN(num):
        return num!= num


    def done(self):
        q,v=self.robot.get_state()
        if math.isnan(v[0]):
            print("***")
            #self.robot.close()
            return True
        return False
