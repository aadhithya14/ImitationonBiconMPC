## This is a demo for trot motion in mpc
## Author : Avadesh Meduri & Paarth Shah
## Date : 21/04/2021

from logging import info
from os import NGROUPS_MAX
import time
import numpy as np
import pinocchio as pin
import tempfile
from bullet_utils.env import BulletEnvWithGround
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from torch import cuda
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
import gym,gym.envs
#import gym_solo
from motions.cyclic.solo12_jump import jump
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

from imitation.algorithms import bc,dagger
#from dagger import SimpleDaggerTrainer
from imitation.algorithms.dagger import SimpleDAggerTrainer
#from imitation.rewards import reward_nets
from imitation.data import rollout
from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import types
from bullet_utils.env import BulletEnvWithGround
import matplotlib.pyplot as plt
from stable_baselines3.common import policies

from imitation.algorithms.adversarial import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
#from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import json

from kinematic_controllers.position_gain_controller import PositionGainController

from termination.base_stability_termination import BaseStabilityTermination
from termination.base_impact_termination import BaseImpactTermination
from termination.knee_impact_termination import KneeImpactTermination
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR




#from stable_baselines3.common.env_util import make_vec_env



class SoloMPC(gym.Env):
    def __init__(self):
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.n_eff = 4
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.des=0.1
        self.x0 = np.concatenate((self.q0, pin.utils.zero(self.pin_robot.model.nv)))
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0)
        self.obs_dim=41
        self.action_dim=12
        self.ndof=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space=Box(-self.high, self.high)
        self.high_action= 3.1415*np.ones([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)
        #np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.num_envs=1 
        self.cnt=0     
        self.reward_list=[] 
        self.termination=Termination(self.robot) 
        self.max_torque=2.5 
        self.joint_limits=[[-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415] ]
        

    def reset(self):
        
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.q_0=self.q0
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        position_old=pybullet.getLinkState(self.robot.robot.robotId,self.robot.robot.end_eff_ids[0])
        #print(position_old)
        #self.q0[0] = np.random.random()
        #self.q0[1] = np.random.random()
        self.q0[0:2] =0.0
        #self.q0[2]=0.35
        #self.q0[6:9]
        #print(self.q0)
        #self.q0[3:6] = np.random.uniform(0.0,0.3)
        self.q0[9] = np.random.uniform(-2.0,-1.0)
        self.q0[12] =self.q0[9]
        self.q0[15] =np.random.uniform(1.0,2.0)
        self.q0[18] = self.q0[15]
        self.q0[8] = np.random.uniform(0.0,1.0)
        self.q0[11]= self.q0[8]
        self.q0[14]= np.random.uniform(-1.0,0.0)
        self.q0[17]= self.q0[14]
        self.robot.robot.reset_state(self.q0, self.v0)
        self.done=False
        """for i in range(5):
            self.qdes_act(self.q_0)"""
        self.des=np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.wdes=np.random.choice([-0.1,-0.05, 0.0, 0.05,0.1])
        
        self.termination.init_termination()
        for termination_object in self.termination.termination_dict.values():
            #print(termination_object)
            termination_object.reset()
        #self.des=np.random.choice([-0.1,-0.05])
        position=pybullet.getLinkState(self.robot.robot.robotId,self.robot.robot.end_eff_ids[0])
        #print(position)


        self.cnt=0
        self.x0 = np.concatenate((self.q0, np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes])))

        #print(self.x0)
        return self.x0

    def step(self,action):
        self.q0,self.v0=self.robot.get_state()
        self.x0=np.concatenate((np.array(self.q0),np.array(self.v0)))
        self.actions(action,action_space="pd")
        
        self.q,self.v=self.robot.get_state()
        
        self.reward_func= Reward(self.v, self.des)
        self.reward= self.reward_func.get_reward()

        self.cnt=self.cnt+1
        kp=2.0
        kd=0.05
        #self.done= False
        """
        if self.cnt>=10000:
            self.done =True
        else:
            self.done= False
        """
        #self.x=np.concatenate((np.array(self.q),np.array(self.v)))
        
        #self.done= False
        for termination_object in self.termination.termination_dict.values():
            if termination_object.done():
                self.done = True
            termination_object.step()
        
        self.x=np.concatenate((np.array(self.q),np.array(self.v),np.array([self.des,0.0,0.0]),np.array([self.wdes])))


        return self.x,self.reward,self.done,info

    def get_robot(self):
        return self.robot

    def render(self,mode="rgb_array"):
        self.close()
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0)
        self.robot.robot.reset_state(self.q0, self.v0)

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pybullet.disconnect(self.robot.env.physics_client)

    def actions(self,action,action_space):
        if action_space=="torque":
            self.robot.send_joint_command(action)
        else:
            self.qdes_act(action)
            #print("**")
    
    def find_qdes(self,tau,kp,kd):
        q,v=self.robot.get_state()
        qdes=np.add(((1/kp)*np.add(np.array(tau),np.array(kd*v[6:]))),q[7:])
        return qdes

    def qdes_act(self,qdes):
        # kp_var (qdes - q) - kd_var qdot

        # example parametrization
        # kp_var_range: [  1.00, 10.00 ]
        # kd_var_range: [  0.05,  0.14 ]

        #action_qdes = action[:self.ndof]
        #action_gain = action[self.ndof:]
        
        kp = 2.0
        kd = 0.05

        

        #self.qdes      = scale(action_qdes, [-1.0, 1.0], self.joint_limits)

        q, v    = self.robot.get_state()
        #self.qdes_for_ttr = self.qdes
        #self.qdotdes_for_ttr = self.qdotdes
        v=v[6:]
        joint_type = ['circular'] * 12
        q_diff     = PositionGainController.pos_diff(q[7:], qdes, joint_type)

        torque = kp * q_diff - kd * v

        self.torque_control(torque)

    def torque_control(self, des_torque, no_clipping=False):
        if not no_clipping:
            self.des_torque = np.clip(des_torque, -self.max_torque, self.max_torque)
        else:
            self.des_torque = des_torque

        

        cont_joint_ids=self.robot.robot.bullet_joint_ids
        """if self.log is not None:
            self.log.add('torque', self.des_torque.tolist())
        """
        for i in range(12):
            pybullet.setJointMotorControl2(
                bodyUniqueId=self.robot.robot.robotId,
                jointIndex=cont_joint_ids[i],
                controlMode=pybullet.TORQUE_CONTROL,
                force=self.des_torque[i])
        pybullet.stepSimulation()

    


# Fucntion to train Behaviour Cloning 

# Expert to sample expert trajectories

class expert_policy():
    def __init__(self,env,exp_name):
        self.v_des = np.array([0.2,0.0,0.0])
        self.w_des = 0.0
        self.obs_dim=41
        self.action_dim=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space = Box(-self.high, self.high)
        self.high_action= 3.145*np.ones([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)

        self.plan_freq = 0.05 # sec
        self.update_time = 0.0 # sec (time of lag)

        self.sim_t = 0.0
        self.sim_dt = .001
        self.index = 0
        self.pln_ctr = 0
        self.env=env
        self.exp_name=exp_name
        ## Motion
        self.gait_params = jump
        self.lag = int(self.update_time/self.sim_dt)

        

        self.plot_time = 0 #Time to start plotting

        self.solve_times = []
        self.trajectories=[]
        #self.obs=np.concatenate([np.array(self.q),np.array(self.v)])
        self.c=0

        pin_robot = Solo12Config.buildRobotWrapper()
        urdf_path = Solo12Config.urdf_path
        x0=self.env.reset()
        #print(x0)
        self.gg = SoloMpcGaitGen(pin_robot,urdf_path,x0,self.plan_freq,x0[0:19], None)


        self.gg.update_gait_params(self.gait_params, self.sim_t)

        f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        self.actions="pd"
        #added
        #X_init = np.zeros(9)
        #X_init[0:3] = pin.centerOfMass(robot.model, robot.data, q0,  pin.utils.zero(robot.model.nv))
        #X_init[4] = 0.5
        #X_ter = X_init.copy()

        #X_nom = np.zeros((9*int(np.round(gait_params.gait_period/gait_params.dt,2))))
        #X_nom[2::9] = X_init[2]

    def predict(self,obs,deterministic=True):
            #rewards=[]
            #reward=0
            # if o == int(50*(plan_freq/sim_dt)):
            #     gait_params = trot
            #     gg.update_gait_params(gait_params, sim_t)
            #     robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)
        self.env.reset()
        if self.pln_ctr == 0:
            contact_configuration = self.env.robot.get_current_contacts()

        self.c+=1    
        pr_st = time.time()
        xs_plan, us_plan, f_plan = self.gg.optimize(obs[0:19], obs[19:], np.round(self.sim_t,3), self.v_des, self.w_des)

                # Plot if necessary
                # if sim_t >= plot_time:
                    # gg.plot_plan(q, v)
                    # gg.save_plan("trot")

        pr_et = time.time()
        self.solve_times.append(pr_et - pr_et)

            # first loop assume that trajectory is planned
        if self.c < int(self.plan_freq/self.sim_dt) - 1:
            xs = xs_plan
            us = us_plan
            f = f_plan

            # second loop onwards lag is taken into account
        elif self.pln_ctr == self.lag and self.c > int(self.plan_freq/self.sim_dt)-1:
                # Not the correct logic
                # lag = int((1/sim_dt)*(pr_et - pr_st))
            lag = 0
            xs = xs_plan[lag:]
            us = us_plan[lag:]
            f = f_plan[lag:]
            index = 0

        tau = self.robot_id_ctrl.id_joint_torques(obs[0:19], obs[19:], xs[index][:self.env.pin_robot.model.nq].copy(), xs[index][self.env.pin_robot.model.nq:].copy()\
                                        , us[index], f[index], contact_configuration)

            #tau=np.array(tau)
            #print(tau)
            
            #env.robot.send_joint_command(tau)
        if self.actions=="torque":
                next_obs,reward,done,info =self.env.step(tau)
        elif self.actions=="increased_frequency_pd":
                action=xs[index+1][7:self.env.pin_robot.model.nq].copy()
                next_obs,reward,done,info =self.env.step(action)
        else:
            qdes=self.env.find_qdes(tau,self.kp,self.kd)
                    #for i in range(8):
            next_obs,reward,done,info =self.env.step(qdes)
                    #print("###")

        self.sim_t += self.sim_dt
        self.pln_ctr = int((self.pln_ctr + 1)%(self.plan_freq/self.sim_dt))
        self.index += 1
            #rewards.append(reward)

            
            # time.sleep(0.001)

            #keys = ["obs", "next_obs", "acts", "dones", "infos"]
            #parts = {key: [] for key in keys}
            #self.trajectory={"obs":obs,"acts":tau}
            #self.trajectories.append(self.trajectory)
        return qdes

    def collect_trajectories(self, trajectories):
        dataset=[]
        for traj in trajectories:
            qdes=self.predict(traj)
            dataset.append({'state':traj,'acts':qdes})
        return dataset

    


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

class Datasplit():
    def __init__(self,expert_observations,expert_actions):
        self.expert_observations=expert_observations
        self.expert_actions=expert_actions
        self.expert_dataset = ExpertDataSet(expert_observations, expert_actions)
        self.train_size = int(0.8 * len(self.expert_dataset))
        self.test_size = len(self.expert_dataset) - self.train_size
        self.train_expert_dataset, self.test_expert_dataset = random_split(
            self.expert_dataset, [self.train_size, self.test_size])

    def __get_train_expert__(self):
        return self.train_expert_dataset

    def __get_test_expert__(self):
        return self.test_expert_dataset


class Reward():
    def __init__(self,vel,des):
        self.vel= vel
        self.des= des

    def get_reward(self):
        self.reward= 5*np.linalg.norm((self.vel[0]-self.des))
        return self.reward


class Termination():
    def __init__(self,robot) -> None:
        self.termination_dict = {}
        self.robot=robot
    
    def init_termination(self):
        #self.termination_dict['imitation_length_termination'] = ImitationLengthTermination(self)
        self.termination_dict['base_stability_termination'] = BaseStabilityTermination(self.robot, max_angle=0.4)
        self.termination_dict['base_impact_termination'] =  BaseImpactTermination(self.robot)
        self.termination_dict['knee_impact_termination'] = KneeImpactTermination(self.robot)


class Agent(): 
    def __init__(self,env,
        student,
        exp_name="/home/aiyer/new_ws",
        batch_size=64,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        test_batch_size=64,
        beta=0.4
       
    ):
        #self.use_cuda = not no_cuda and torch.cuda.is_available()
        torch.manual_seed(seed)
        self.env=env
        self.student=student
        self.batch_size=batch_size
        #self.epochs=epochs
        self.exp_name=exp_name
        exp_config = yaml.load(open(exp_name + 'conf.yaml'))
        self.epochs=exp_config['n_epochs']
        self.scheduler_gamma=scheduler_gamma
        self.learning_rate=learning_rate
        self.log_interval=log_interval
        self.test_batch_size=test_batch_size
        self.beta=beta
        self.device = torch.device("cpu")
        self.plotlog=Log(self.exp_name+"losses.json")

        self.expert=expert_policy(self.env,self.exp_name)

        self.kwargs = {"num_workers": 1, "pin_memory": True} 

        if isinstance(env.action_space, gym.spaces.Box):
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        self.model = student.policy.to(self.device)

    def train(self, device, train_loader, optimizer,epoch):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                if isinstance(self.env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                    if isinstance(self.student, (A2C, PPO)):
                        action, _, _ = self.model(data)
                    else:
                        # SAC/TD3:
                        action = self.model(data)
                    action_prediction = action.double()
                else:
                # Retrieve torche logits for A2C/PPO when using discrete actions
                    dist = self.model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                loss = self.criterion(action_prediction, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                    if batch_idx==0:
                        self.plotlog.add('loss',[loss.item()])
            self.plotlog.save()             
    def test(self, device, test_loader):
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    if isinstance(self.env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                        if isinstance(self.student, (A2C, PPO)):
                            action, _, _ = self.model(data)
                        else:
                            # SAC/TD3:
                            action = self.model(data)
                        action_prediction = action.double()
                    else:
                        # Retrieve torche logits for A2C/PPO when using discrete actions
                        dist = self.model.get_distribution(data)
                        action_prediction = dist.distribution.logits
                        target = target.long()

                    test_loss = self.criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    def data_aggregator(self):
        trajectories=[]
        if self.beta>0.3:
            for i in range(100):
                state=self.env.reset()
                action=self.expert.predict(state)
                next_state,reward,done,_=self.env.step(action)
                trajectories.append(state)
        else:
            for i in range(100):
                state=self.env.reset()
                action=self.student.policy.predict(state)
                next_state,reward,done,_=self.env.step(action)
                trajectories.append(state)

        return trajectories

    def merge_dataset(self,dataset_old,dataset_new):
        dataset_old.append(dataset_new)
        return dataset_old

    def get_observation_and_action(self,trajectories):
        observations_list=[]
        actions_list=[]
        for traj in trajectories:
            observations=traj['state']
            observations_list.append(observations)
            actions=traj['state']
            actions_list.append(actions)
        return observations_list,actions_list

    def dagger_play(self):
        dataset_old=[]
        for i in range(100000):
            trajectories=self.data_aggregator(self.student,self.beta)
            dataset_new=self.expert.collect_trajectories(trajectories)
            dataset_old=self.merge_dataset(dataset_old,dataset_new)
            observations,actions=self.get_observation_and_action(dataset_old)
            data=Datasplit(observations,actions)
            train_expert_dataset=data.__get_train_expert__()
            test_expert_dataset=data.__get_test_expert__()
            self.play(train_expert_dataset,test_expert_dataset)

            
    def play(self,train_expert_dataset,test_expert_dataset):
            train_loader = torch.utils.data.DataLoader(
                dataset=train_expert_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_expert_dataset, batch_size=self.test_batch_size, shuffle=True, **self.kwargs,
            )

            # Define an Optimizer and a learning rate schedule.
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=self.scheduler_gamma)

            # Now we are finally ready to train torche policy model.
            for epoch in range(1, self.epochs + 1):
                self.train(self.device, train_loader, optimizer,epoch)
                #test(model, device, test_loader)
                scheduler.step()

            # Implant torche trained policy network back into the RL student agent
            a2c_student.policy = self.model
   

def visualise(venv,student,n_eval_episodes=3,render=False,exp_name="/home/aiyer/new_ws"):
        policy=student.load(exp_name+"a2c_student")
        #venv.close()

        reward, _ = evaluate_policy(policy, venv, n_eval_episodes=n_eval_episodes, render=True)
        print(f"Reward after training: {reward}") 


if __name__== "__main__":

    env_id = 'Solo12-v1'
    import argparse
    import yaml
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Solo12-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='experiment name', default='unnamed')
    parser.add_argument('--exp', help='')
    args = parser.parse_args()

    
    exp_name= args.exp
    #venv=gym.make(env_id)
    #env=make_vec_env(venv)
    env= SoloMPC() 
    a2c_student = A2C('MlpPolicy', env, verbose=1)
    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    agent= Agent(env,a2c_student,exp_name)
    agent.dagger_play()
    """
    expert_observations,expert_actions=expert(env,exp_name)
    data=Datasplit(expert_observations,expert_actions)
    train_expert_dataset=data.__get_train_expert__()
    test_expert_dataset=data.__get_test_expert__()

    mean_reward, std_reward = evaluate_policy(a2c_student, env, n_eval_episodes=10)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")

    agent= Agent(env,a2c_student,exp_name)
    agent.play(train_expert_dataset,test_expert_dataset)

    
    a2c_student.save(exp_name+"a2c_student")

    mean_reward, std_reward = evaluate_policy(a2c_student, env, n_eval_episodes=10)
   
    print(f"Mean reward = {mean_reward} +/- {std_reward}")
    """
    #venv =  DummyVecEnv([SoloMPC])
    #venv=gym.make(env_id)
    #print(venv.envs[0])
    #env=venv.env
    #print(env.pin_robot)
    #train_bc(venv,exp_name)
    #collect_dataset(venv,exp_name)
    #train_dagger(venv)
    #train_gail(venv)
    #visualise(env,a2c_student,10,True,exp_name)

    

