## This is a demo for trot motion in mpc
## Author : Avadesh Meduri & Paarth Shah
## Date : 21/04/2021

from logging import info
import time
import numpy as np
import pinocchio as pin
import tempfile
from bullet_utils.env import BulletEnvWithGround
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from torch import cuda
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
from utils.my_math import scale
import gym,gym.envs
#import gym_solo
from motions.cyclic.solo12_jump import jump
from motions.cyclic.solo12_trot import trot
from motions.cyclic.solo12_bound import bound
from gym.spaces import Box
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.cmd_util import make_vec_env

from biconvex_mpc_cpp import BiconvexMP
import pybullet
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
from imitation.algorithms.bc import reconstruct_policy

from imitation.algorithms import bc
#from dagger import SimpleDaggerTrainer
#from imitation.algorithms.dagger import SimpleDAggerTrainer
#from imitation.rewards import reward_nets
from imitation.data import rollout
from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import types
from bullet_utils.env import BulletEnvWithGround
import matplotlib.pyplot as plt
from stable_baselines3.common import policies

#from imitation.algorithms.adversarial import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
#from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import json
from stable_baselines3 import PPO, A2C, SAC, TD3
from kinematic_controllers.position_gain_controller import PositionGainController

from termination.base_stability_termination import BaseStabilityTermination
from termination.base_impact_termination import BaseImpactTermination
from termination.knee_impact_termination import KneeImpactTermination
from termination.solver_termination import SolverTermination

from envs.gymenv import SoloMPC
from Expert.expert_mpc import Expert
from rewards.Velocityreward import Reward
#from termination.termination import Termination
from stable_baselines3.common.policies import ActorCriticPolicy

#from stable_baselines3.common.env_util import make_vec_env

#Constant Learning Rate schedule

class ConstantLRSchedule:
    """A callable that returns a constant learning rate."""

    def __init__(self, lr: float = 1e-3):
        """
        Args:
            lr: the constant learning rate that calls to this object will return.
        """
        self.lr = lr

    def __call__(self, _):
        """
        Returns the constant learning rate.
        """
        return self.lr

#Imitation Learning agent

class Agent():
    def __init__(self,env,student,exp_name,collect):
        self.env=env
        self.student=student
        self.exp_name=exp_name
        self.collect=collect

    #Function to train BC

    def train_bc(self):
        if self.collect==1:
            self.collect_transitions()
        else:
            exp_config = yaml.load(open(self.exp_name + 'conf.yaml'))
            n_epochs=exp_config['n_epochs']
            #transitions=sample_expert_transitions(env)
            log=json.load(open(exp_name+"obs_new.json"))
            transitions=[]
            for i in range(320000):
                transitions.append({'obs':np.array(log['obs'][i]),'acts':np.array(log['acts'][i])})
            bc_trainer = bc.BC(
                    observation_space=self.env.observation_space,
                    action_space=self.env.action_space,
                    expert_data=transitions,
                    student=self.student
            )
     
            
            #reward_before, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=True)
            #print(f"Reward before training: {reward_before}")

            bc_trainer.train(n_epochs,exp_name=exp_name)
            bc_trainer.save_policy(exp_name+"policy")

            reward_after, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=20, render=False)
            print(f"Reward after training: {reward_after}")

            
   
    #Function  to train dagger policy

    def train_dagger(self):
        transitions=expert(env,exp_name)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            expert_data=transitions,
            student=student
        )
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                env=env, scratch_dir=tmpdir, expert_policy=expert, bc_trainer=bc_trainer,exp_name=exp_name
            )
            #reward, _ = evaluate_policy(dagger_trainer.policy, env, n_eval_episodes=3, render=False)
            #print(f"Reward before training: {reward}")
            dagger_trainer.train(200000)
            dagger_trainer.save_policy(exp_name+"policy")

        reward, _ = evaluate_policy(dagger_trainer.policy, env, n_eval_episodes=3, render=False)
        print(f"Reward after training: {reward}")

    #Function to train GAIL 

    def train_gail(self):
        
        transitions= expert(env)
        learner = PPO(
                env=venv,
                policy=MlpPolicy,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0003,
                n_epochs=10,
                )
        """reward_net = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
        )"""
        gail_trainer = GAIL(
            venv=venv,
            expert_data=transitions,
            expert_batch_size=1024,
            n_disc_updates_per_round=4,
            gen_algo=learner,
            
        )
        learner_rewards_before_training, _ = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        )

        print(learner_rewards_before_training)
        gail_trainer.train(20000)  # Note: set to 300000 for better results
        learner_rewards_after_training, _ = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        )

    #Function to run mpc

    def collect_transitions(self):
        expert=Expert(self.env,self.exp_name)
        transitions=expert.rollout()
        return transitions

    #Function to visualise policy 

    def visualise(self,n_eval_episodes=10,render=False):
        policy=reconstruct_policy(self.exp_name+"/policy")
        #obs = env.reset()   
        #venv.close()
        #logging_id=pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4,exp_name+"video.mp4")
        reward, _ = evaluate_policy(policy, self.env,self.exp_name, n_eval_episodes=n_eval_episodes)
            #pybullet.stopStateLogging(logging_id)
            #print(f"Reward after training: {reward}"


if __name__== "__main__":

    env_id = 'Solo12-v1'
    import argparse
    import yaml
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Solo12-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='experiment name', default='unnamed')
    parser.add_argument('--flag', help='experiment name',type=int, default='unnamed')
    parser.add_argument('--exp', help='')
    parser.add_argument('--plot', help='',type=int)
    parser.add_argument('--collect', help='',type=int)
    args = parser.parse_args()

    
    exp_name= args.exp
    flag=args.flag
    plot=args.plot
    collect=args.collect
    print(flag)
    #venv=gym.make(env_id)
    #env=make_vec_env(venv)
    env= SoloMPC(exp_name,flag,plot) 

    #venv =  DummyVecEnv([SoloMPC])
    #venv=gym.make(env_id)
    #print(venv.envs[0])
    #env=venv.env
    #print(env.pin_robot)
    print(env.action_space)
    """policy_kwargs = dict(
            lr_schedule=ConstantLRSchedule(lr=0.0003),
            #device=self.device,
        )"""
    a2c_student = A2C(ActorCriticPolicy, env, verbose=1)
    agent= Agent(env,a2c_student,exp_name,collect)
    if flag==0:
        agent.train_bc()
        
    else:
        agent.visualise(n_eval_episodes=5)
    #collect_dataset(venv,exp_name)
    #train_dagger(env,a2c_student,exp_name)
    #train_gail(venv)
   

    

