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

def expert(env,exp_name):

    exp_config = yaml.load(open(exp_name + 'conf.yaml'))
    w_des = 0.0

    plan_freq = 0.05 # sec
    update_time = 0.0 # sec (time of lag)

    sim_t = 0.0
    sim_dt = .001
    index = 0
    pln_ctr = 0

    x0=env.reset()
    x0=x0[0:37]
    ## Motion
    
    gait_params = jump
    lag = int(update_time/sim_dt)
    gg = SoloMpcGaitGen(env.pin_robot, env.urdf_path, x0, plan_freq, env.q0, None)

    gg.update_gait_params(gait_params, sim_t)

    #robot = PyBulletEnv(Solo12Robot, q0, v0)
    robot_id_ctrl = InverseDynamicsController(env.pin_robot, env.f_arr)
    robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

    plot_time = 0 #Time to start plotting
    trajectory_list=[]
    action_space="pd"
    kp=2.0
    kd=0.05
    solve_times = []
    num_interactions= int(1e6)
    j=0
    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + (int(env.observation_space.shape[0]),))
        expert_actions = np.empty((num_interactions,) + (int(env.action_space.shape[0]),))

    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)
    
    #log=Log(exp_name+"data.json")

    #trajectories= {"obs":[[x0]],"acts":[],"rews":[], "infos":[]}
    for i in range(exp_config['timesteps']):
        x0=env.reset()
        v_des = np.array([env.des,0.0,0.0])
        w_des = 0.0
        trajectories=[]
        env.qdes_act(env.q0[7:])
        for i in range(5):
            pybullet.stepSimulation()
        #log.save()
        fobs=[]
        freward=[]
        ftau=[]
        finfos=[]
        done=False
        qdes_list=[]
        obs_list=[]
        for o in range(int(exp_config['steps']*(plan_freq/sim_dt))):
                # this bit has to be put in shared memory
            #if not done:
                q, v = env.robot.get_state()
                obs=np.concatenate((np.array(q),np.array(v),v_des,np.array([w_des])))
                    #obs=obs.tolist()
                    #trajectories["obs"]=obs
                    
                    # if o == int(50*(plan_freq/sim_dt)):
                    #     gait_params = trot
                    #     gg.update_gait_params(gait_params, sim_t)
                    #     robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

                if pln_ctr == 0:
                    contact_configuration = env.robot.get_current_contacts()
                        
                    pr_st = time.time()
                    xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t,3), v_des, w_des)

                        # Plot if necessary
                        # if sim_t >= plot_time:
                            # gg.plot_plan(q, v)
                            # gg.save_plan("trot")

                    pr_et = time.time()
                    solve_times.append(pr_et - pr_et)

                    # first loop assume that trajectory is planned
                if o < int(plan_freq/sim_dt) - 1:
                    xs = xs_plan
                    us = us_plan
                    f = f_plan

                    # second loop onwards lag is taken into account
                elif pln_ctr == lag and o > int(plan_freq/sim_dt)-1:
                        # Not the correct logic
                        # lag = int((1/sim_dt)*(pr_et - pr_st))
                    lag = 0
                    xs = xs_plan[lag:]
                    us = us_plan[lag:]
                    f = f_plan[lag:]
                    index = 0

                tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:env.pin_robot.model.nq].copy(), xs[index][env.pin_robot.model.nq:].copy()\
                                                , us[index], f[index], contact_configuration)

                #next_obs,reward,done,info =env.step(tau)

                
                if action_space=="torque":
                    next_obs,reward,done,info =env.step(tau)
                elif action_space=="increased_frequency_pd":
                    action=xs[index+1][7:env.pin_robot.model.nq].copy()
                    next_obs,reward,done,info =env.step(action)
                else:
                    qdes=env.find_qdes(tau,kp,kd)
                    #for i in range(8):
                    next_obs,reward,done,info =env.step(qdes)
                    #print("###")


                if done:
                    break

                sim_t += sim_dt
                pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
                index += 1
                
                qdes_list.append(qdes)
                obs_list.append(obs)
        if not done:
            for k in range(len(qdes_list)):
                expert_actions[j]=qdes_list[k]
                expert_observations[j]=obs_list[k]
                j=j+1
                #trajectory={'obs':obs,'acts':qdes,'rews':reward,'infos':[0]}
                #trajectories.append(trajectory)
                #print(done)

    np.savez_compressed(
            exp_name+"expert_data",
            expert_actions=expert_actions[:j],
            expert_observations=expert_observations[:j])

    return expert_observations[:j],expert_actions[:j]
    


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
        
        self.device = torch.device("cpu")
        self.plotlog=Log(self.exp_name+"losses.json")

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

            # Now we are finally ready to train torch policy model.
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
    visualise(env,a2c_student,10,True,exp_name)

    

