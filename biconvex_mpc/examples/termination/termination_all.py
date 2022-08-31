from termination.base_stability_termination import BaseStabilityTermination
from termination.base_impact_termination import BaseImpactTermination
from termination.knee_impact_termination import KneeImpactTermination
from termination.solver_termination import SolverTermination

class Termination():
    def __init__(self,robot) -> None:
        self.termination_dict = {}
        self.robot=robot
    
    def init_termination(self):
        
        #self.termination_dict['imitation_length_termination'] = ImitationLengthTermination(self)
        self.termination_dict['base_stability_termination'] = BaseStabilityTermination(self.robot, max_angle=0.6)
        self.termination_dict['base_impact_termination'] =  BaseImpactTermination(self.robot)
        self.termination_dict['knee_impact_termination'] = KneeImpactTermination(self.robot)
        self.termination_dict['solver_termination']= SolverTermination(self.robot)