import gym
from logging import info

from bullet_utils.env import BulletEnvWithGround
import pinocchio as pin
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
import numpy as np
from gym.spaces import Box
import pybullet
from envs.pybullet_env import PyBulletEnv

class SoloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.n_eff = 4
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.x0 = np.concatenate([self.q0, pin.utils.zero(self.pin_robot.model.nv)])
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0)
        self.obs_dim=37
        self.action_dim=12
        self.ndof=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space=Box(-self.high, self.high)
        self.high_action= np.ones([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)
        self.des=0.1#np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.num_envs=1 
        self.cnt=0     
        self.reward_list=[]   
        

    def reset(self):
        
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.n_eff = 4
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.x0 = np.concatenate([self.q0, pin.utils.zero(self.pin_robot.model.nv)])
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0)
        self.obs_dim=37
        self.action_dim=12
        self.ndof=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space=Box(-self.high, self.high)
        self.high_action= np.ones([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)
        self.des=0.1#np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.num_envs=1 
        self.cnt=0     
        self.reward_list=[]   
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.cnt=0
        self.q0[0:2] = 0.0
        self.x0 = np.concatenate([self.q0, pin.utils.zero(self.pin_robot.model.nv)])
        return self.x0

    def step(self,action):
        self.q0,self.v0=self.robot.get_state()
        self.x0=np.concatenate((np.array(self.q0),np.array(self.v0)))
        #self.reward=1 
        #self.reward = self.reward_net.forward(self.x,action,self.self.done)
        #self.reward= -self.loss
        print(action)
        self.robot.send_joint_command(action)
        self.q,self.v=self.robot.get_state()
        
        self.reward_func= Reward(self.v, self.des)
        self.reward= self.reward_func.get_reward()

        self.cnt=self.cnt+1
        if self.cnt>=5:
            self.done =[True]
        else:
            self.done= [False]
        self.x=np.concatenate((np.array(self.q),np.array(self.v)))
        
        return self.x.tolist(),self.reward,self.done,info

    def render(self,mode="rgb_array"):
        env=BulletEnvWithGround(server=pybullet.GUI)
        robot=env.add_robot(Solo12Robot)
        robot.reset_state(self.q0, self.v0)

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pybullet.disconnect(self.robot.env.physics_client)


class Reward():
    def __init__(self,vel,des):
        self.vel= vel
        self.des= des

    def get_reward(self):
        self.reward= 5*np.linalg.norm((self.vel[0]-self.des))
        return self.reward
