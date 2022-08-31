import numpy as np


class KneeImpactTermination():

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        contacts = self.robot.p.getContactPoints(
                       bodyA=self.robot.env.objects[0],
                       bodyB=self.robot.robot.robotId)
        
        for contact in contacts:
            if contact[4] in self.robot.get_upper_leg_link_ids():
                return True
        
        return False
