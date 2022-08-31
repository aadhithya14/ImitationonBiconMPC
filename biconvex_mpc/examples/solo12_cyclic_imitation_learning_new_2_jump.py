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
import gym,gym.envs
from utils.my_math import scale
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
from stable_baselines3 import PPO, A2C, SAC, TD3
from kinematic_controllers.position_gain_controller import PositionGainController

from termination.base_stability_termination import BaseStabilityTermination
from termination.base_impact_termination import BaseImpactTermination
from termination.knee_impact_termination import KneeImpactTermination
from termination.solver_termination import SolverTermination




#from stable_baselines3.common.env_util import make_vec_env


#Gym Environment
class SoloMPC(gym.Env):
    def __init__(self,exp_name,flag):
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.n_eff = 4
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        #self.des=0.0
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
        self.exp_name=exp_name
        self.exp_config=yaml.load(open(exp_name + 'conf.yaml'))
        #np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.num_envs=1
        self.flag=flag 
        self.cnt=0     
        self.reward_list=[] 
        self.termination=Termination(self.robot) 
        self.max_torque=2.5 
        self.joint_limits=[[-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415] ]
        self.plotlog=Log(exp_name+"actions.json")
        self.failure=0
        

    def reset(self):
        
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.q_0=self.q0
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        position_old=pybullet.getLinkState(self.robot.robot.robotId,self.robot.robot.end_eff_ids[0])
        #print(position_old)
        #self.q0[0] = np.random.random()
        #self.q0[1] = np.random.random()
        self.q0[0:2] =0.0
        #self.q0[2]=np.random.uniform(0.35,0.5)
        #self.q0[6:9]
        #print(self.q0)
        #self.q0[3:6] = np.random.uniform(0.0,0.05)
        """self.q0[7] = np.random.uniform(-0.2,0.2)
        self.q0[10] = np.random.uniform(-0.2,0.2)
        self.q0[13] = np.random.uniform(-0.2,0.2)
        self.q0[16] = np.random.uniform(-0.2,0.2)
        

        self.q0[8] = np.random.uniform(0.6,1.5)
        self.q0[11] = np.random.uniform(0.6,1.5)
        self.q0[14] =  np.random.uniform(-1.5,-0.6)
        self.q0[17] =  np.random.uniform(-1.5,-0.6)
        self.q0[9] = np.random.uniform(-2.0,-1.4)
        self.q0[12]= np.random.uniform(-2.0,-1.4)
        self.q0[15]= np.random.uniform(1.4,2.0)
        self.q0[18]= np.random.uniform(1.4,2.0)
        """
        """self.q0[7] = np.random.uniform(-0.1,0.1)
        self.q0[10] = np.random.uniform(-0.1,0.1)
        self.q0[13] = np.random.uniform(-0.1,0.1)
        self.q0[16] = np.random.uniform(-0.1,0.1)
        

        self.q0[8] = np.random.uniform(0.6,1.2)
        self.q0[11] = np.random.uniform(0.6,1.2)
        self.q0[14] =  np.random.uniform(-1.2,-0.6)
        self.q0[17] =  np.random.uniform(-1.2,-0.6)
        self.q0[9] = np.random.uniform(-2.0,-1.4)
        self.q0[12]= np.random.uniform(-2.0,-1.4)
        self.q0[15]= np.random.uniform(1.4,2.0)
        self.q0[18]= np.random.uniform(1.4,2.0)
        """        
        #euler=list(pybullet.getEulerFromQuaternion([self.q0[3],self.q0[4],self.q0[5],self.q0[6]]))
        #print(euler)
        #print(euler[0:2])
        #euler[0]+=np.random.uniform(-0.2,0.2)
        #euler[1]+=np.random.uniform(-0.2,0.2)
        #quaternion=pybullet.getQuaternionFromEuler(euler)
        #self.q0[3:7]=quaternion
        self.robot.robot.reset_state(self.q0, self.v0)
        self.done=False
        """for i in range(5):
            self.qdes_act(self.q_0)"""
        #self.des=np.random.uniform(-0.2,0.2)
        #self.wdes=np.random.uniform(-0.1,0.1)
        #self.des=np.random.uniform(-0.2,0.2)
        self.des=0.0
        self.wdes=0.0
       
        
        self.termination.init_termination()
        for termination_object in self.termination.termination_dict.values():
            #print(termination_object)
            termination_object.reset()
        
        #self.des=np.random.choice([-0.1,-0.05])
        position=pybullet.getLinkState(self.robot.robot.robotId,self.robot.robot.end_eff_ids[0])
        #self.plotlog.save()
        #print(position)
        

        self.cnt=0
        if self.exp_config['state_conf']=='no_x_y':
            self.q0[0]=0.0
            self.q0[1]=0.0
        elif self.exp_config['state_conf']=='no_yaw_no_x_y':
            self.q0[0]=0.0
            self.q0[1]=0.0
            self.q0[3]=quaternion[0]
            self.q0[4]=quaternion[1]
            self.q0[5]=quaternion[2]
            self.q0[6]=quaternion[3]
        
        self.x0 = np.concatenate((self.q0, np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes])))

        #print(self.x0)
        return self.x0

    def step(self,action):


        self.q0,self.v0=self.robot.get_state()
        self.x0=np.concatenate((np.array(self.q0),np.array(self.v0)))
        self.actions(action,action_space="pd")
        
        self.q,self.v=self.robot.get_state()
        #print(self.v[0])
       
        #self.plotlog.add('actual_qdes',self.v[0])
        #self.plotlog.add('qdes_command',self.des)
        
        
        self.reward_func= Reward(self.v, self.des)
        self.reward= self.reward_func.get_reward()

        self.cnt=self.cnt+1
        kp=2.0
        kd=0.05
        
        if self.flag!=0:
            #self.done=False
            if self.cnt>=10000:
                self.done =True
            else:
                self.done= False
           
                #self.x=np.concatenate((np.array(self.q),np.array(self.v)))
        else: 
            self.done= False
            for termination_key,termination_object in self.termination.termination_dict.items():
                        if termination_object.done():
                            if self.cnt>=300:
                                self.done = True
                            if termination_key=='solver_termination':
                                self.done=True
                                self.render()
                        termination_object.step()
        

        if self.done:
            self.failure=self.failure+1
        
        euler=list(pybullet.getEulerFromQuaternion([self.q[3],self.q[4],self.q[5],self.q[6]]))
        euler[2]=0.0
        quaternion=pybullet.getQuaternionFromEuler(euler)


        if self.exp_config['state_conf']=='no_x_y':
            self.q[0]=0.0
            self.q[1]=0.0
        

        elif self.exp_config['state_conf']=='no_yaw_no_x_y':
            self.q[0]=0.0
            self.q[1]=0.0
            self.q[3]=quaternion[0]
            self.q[4]=quaternion[1]
            self.q[5]=quaternion[2]
            self.q[6]=quaternion[3]
        
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

    #Functions to get the actions corresponding to type of action space
    def actions(self,action,action_space):
        if action_space=="torque":
            self.robot.send_joint_command(action)
        else:
            self.qdes_act(action)
            #print("**")
    
    #Function to Find the qdes corresponding to the torques obtained from MPC
    def find_qdes(self,tau,kp,kd):
        q,v=self.robot.get_state()
        qdes=np.add(((1/kp)*np.add(np.array(tau),np.array(kd*v[6:]))),q[7:])
        return qdes

    #PositionGainController
    def qdes_act(self,qdes):
        # kp_var (qdes - q) - kd_var qdot

        # example parametrization
        # kp_var_range: [  1.00, 10.00 ]
        # kd_var_range: [  0.05,  0.14 ]

        #action_qdes = action[:self.ndof]
        #action_gain = action[self.ndof:]
        
        kp = 2.0
        kd = 0.05

        

        #self.qdes      = scale(qdes, [-1.0, 1.0], self.joint_limits)

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

    

# Fucntion to collect dataset

def collect_dataset(env,exp_name):
    expert(env,exp_name)
    
    

def train_bc(env,student,exp_name):
    exp_config = yaml.load(open(exp_name + 'conf.yaml'))
    n_epochs=exp_config['n_epochs']
    #transitions=sample_expert_transitions(env)
    
    transitions=expert(env,exp_name)
    #print(transitions)
    #print(exp_name.split("/000")[0])

    #for i in range(exp_config["timesteps"]):
    
    """with open(exp_name+"data.json", 'r') as f:
        data= json.load(f)
        #transitions=np.zeros(len(data['rews']))
        #print(data['obs'])
        for i in range(len(data['rews'])):
            #print(data['rews'])
            dictionary={'obs': (data['obs'])[i],'acts': (data['acts'])[i], 'rews': (data['rews'])[i],'infos': (data['infos'])[i]}
            new_list=[dictionary]
            #print(dictionary)
            transitions.extend(new_list)
    """
    
    #transitions=transitions.tolist()
    """bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    expert_data=transitions
    )"""
    

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_data=transitions,
        student=student
        #policy_class=exp_config['policy_class'],
        #policy_kwargs={'net_arch':exp_config['net_arch'],'activation_fn':exp_config['activation_fn']},
        #optimizer_cls= exp_config["optimiser_cls"],

    )
    bc_trainer.train(n_epochs,exp_name=exp_name)
    bc_trainer.save_policy(exp_name+"policy")


    """reward_before, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=True)
    print(f"Reward before training: {reward_before}")

    bc_trainer.train(n_epochs,exp_name=exp_name)
    bc_trainer.save_policy(exp_name+"policy")

    reward_after, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=20, render=False)
    print(f"Reward after training: {reward_after}")
    x = np.arange(5)

    plt.plot(x,reward_before*np.ones(5), "-b", label="before")
    plt.plot(x,reward_after*np.ones(5), "-r", label="after")
    plt.legend(loc="upper left")
    plt.savefig(exp_name+"fig.png")
    """
    #plt.plot(reward_before,leg)

#Function  to train dagger policy

def train_dagger(env,student,exp_name):
    import tempfile

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



#Function to train gail 

def train_gail(env):

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


#Function to visualise policy 

def visualise(venv,n_eval_episodes=3,render=False,exp_name="/home/aiyer/new_ws"):
        policy=reconstruct_policy(exp_name+"policy")
        #venv.close()
        #logging_id=pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4,exp_name+"video.mp4")
        reward, _ = evaluate_policy(policy, venv, n_eval_episodes=n_eval_episodes, render=True)
        done=False
        state= None
        #obs=env.reset()
        
        #network_input= np.array([ 0. ,      0.   ,       0.25 ,       0. ,         0.,          0.,1.    ,     -0.09019157,  0.60605045, -1.68128777, -0.11716085,  0.9512721,-1.93047477, -0.15360379, -1.47684297,  1.41114988,  0.12394247, -0.78277184, 1.83046571,  0.     ,     0.   ,       0.,          0.,          0., 0.  ,        0.  ,        0.    ,      0.   ,       0.     ,     0.,0.    ,      0.    ,      0. ,         0. ,         0. ,         0., 0 ,         0.0837397,   0.,          0.,          0.   ])
        #while not done:
        #print(len(network_input))
        #action, state = policy.predict(network_input, state=state, deterministic=True)
        #print(action)
        #obs, reward, done, _info = env.step(action)
            #episode_reward += reward
        #pybullet.stopStateLogging(logging_id)
        #print(f"Reward after training: {reward}")


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

   
    #state=np.zeros(2)
    #x_old=env.reset()
    x0=env.reset()
    x0=x0[0:37]
    ## Motion
    
    gait_params = jump
    lag = int(update_time/sim_dt)
    gg = SoloMpcGaitGen(env.pin_robot, env.urdf_path, x0, plan_freq,x0[0:19], None)

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
    failure=0
    done=False
    
    log=Log(exp_name+"torque.json")
    #mylog=Log(exp_name+"values_new.json")
    #log=json.load(open(exp_name+"torque.json"))
    #trajectories= {"obs":[[x0]],"acts":[],"rews":[], "infos":[]}
    #logging_id=pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4,exp_name+"video.mp4")
    for i in range(exp_config['timesteps']):
        #x0=env.reset()
        #state=np.zeros(2)
        #x_old=env.reset()
        #x0=np.concatenate((state,x_old))
        x0=env.reset()
        v_des = np.array([x0[37],0.0,0.0])
        x0=x0[0:37]
        
        w_des = 0.0
        trajectories=[]
        for i in range(500):
            env.qdes_act(x0[7:19])
            pybullet.stepSimulation()
              
        #log.save()
        if done==True:
            print("***")
            failure+=1
        done=False
        plan_freq = 0.05 # sec
        update_time = 0.0 # sec (time of lag)
        

        sim_t = 0.0
        sim_dt = .001
        index = 0
        pln_ctr = 0
        lag = int(update_time/sim_dt)
        gg = SoloMpcGaitGen(env.pin_robot, env.urdf_path, x0, plan_freq,x0[0:19], None)

        gg.update_gait_params(gait_params, sim_t)
        for o in range(int(exp_config['steps']*(plan_freq/sim_dt))):
                # this bit has to be put in shared memory
            #if not done:
                q, v = env.robot.get_state()
                obs=np.concatenate((np.array(q),np.array(v),v_des,np.array([w_des])))
                #obs=log['obs'][2500+o]
                #print(obs)
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
                    #mylog.add('qdes',qdes.tolist())
                    #policy=reconstruct_policy(exp_name+"policy")
                    #action,_=policy.predict(obs)
                    #mylog.add('qdes_policy',action.tolist())
                    obs,reward,done,info =env.step(qdes)
                    #print("###")

                sim_t += sim_dt
                pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
                index += 1

                if done:
                    print("###")
                    #pybullet.stopStateLogging(logging_id)
                    break

                


                trajectory={'obs':obs,'acts':qdes,'rews':reward,'infos':[0]}
                trajectories.append(trajectory)
                #print(done)

                """if done:
                    break"""

        if not done:
            trajectory_list.extend(trajectories)
            for traj in trajectories:
                log.add('obs',traj['obs'].tolist())

            """fobs.append(obs.tolist())
            ftau.append(tau.tolist())
            freward.append(reward) 
            finfos.append(0)
        if not done:
            log.add('obs',fobs)    
            log.add('acts',ftau)
            log.add('rews',freward)
            log.add('infos',finfos)"""

        
            
            
                    #trajectories=make_trajectories(trajectory, trajectories)
                    #json.load("")
                    #trajectory_list+=dicts
        """ fobs.append(obs.tolist())
            ftau.append(tau.tolist())
            freward.append(reward) 
            finfos.append(0)
        if not done:
            for i in range(len(fobs)):
                trajectory={'obs':fobs[i],'acts':ftau[i],'rews':freward[i], 'infos':finfos[i]}
                trajectory_list.append(trajectory)
                #trajectory={"obs":[obs],"acts":[tau.tolist()],"rews":[reward], "infos":[[0]]}
                #print(tau)
                #trajectories=make_trajectories(trajectory, trajectories) """
        
            
        #
    #print(len(trajectories["obs"]))
    #make_trajectories_numpy_array(trajectories)
    #traj = types.TrajectoryWithRew(**trajectories)
    
    #assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1
    #trajectories=[trajectories]
    #print(len(trajectory_list))
    #pybullet.stopStateLogging(logging_id)
    #mylog.save()
    log.save()
    print(failure)
    return trajectory_list 


    

def make_trajectories(trajectory,trajectories):
    for key, arr_list in trajectory.items():
        #trajectories[key]=[trajectories[key]]
        print(key)
        print(arr_list)
        trajectories[key].extend(
                arr_list
            )
    return trajectories

def make_trajectories_numpy_array(trajectories):
    for key, arr_list in trajectories.items():
        #trajectories[key]=[trajectories[key]]
        if key=="rews":
            trajectories[key] = np.array(arr_list).flatten()
        else:
            trajectories[key] = np.array(arr_list)
            
    return trajectories

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
        self.termination_dict['base_stability_termination'] = BaseStabilityTermination(self.robot, max_angle=0.5)
        self.termination_dict['base_impact_termination'] =  BaseImpactTermination(self.robot)
        self.termination_dict['knee_impact_termination'] = KneeImpactTermination(self.robot)
        self.termination_dict['solver_termination']= SolverTermination(self.robot)


   




    

class expert_policy():
    def __init__(self,obs,env):
        self.v_des = np.array([0.2,0.0,0.0])
        self.w_des = 0.0
        self.obs_dim=37
        self.action_dim=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space = Box(-self.high, self.high)
        self.high_action= np.ones([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)

        self.plan_freq = 0.01 # sec
        self.update_time = 0.0 # sec (time of lag)

        self.sim_t = 0.0
        self.sim_dt = .001
        self.index = 0
        self.pln_ctr = 0
        self.env=env
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
        print(x0)
        self.gg = SoloMpcGaitGen(pin_robot,urdf_path,x0,self.plan_freq,x0[0:19], None)


        self.gg.update_gait_params(self.gait_params, self.sim_t)

        f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        
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
        next_state,reward,done, info=self.env.step(tau)

            
            #rewards.append(reward)

            
            # time.sleep(0.001)
        self.sim_t += self.sim_dt
        self.pln_ctr = int((self.pln_ctr + 1)%(self.plan_freq/self.sim_dt))
        index += 1

            #keys = ["obs", "next_obs", "acts", "dones", "infos"]
            #parts = {key: [] for key in keys}
            #self.trajectory={"obs":obs,"acts":tau}
            #self.trajectories.append(self.trajectory)
        return tau
        #return self.trajectories     
    



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
    args = parser.parse_args()

    
    exp_name= args.exp
    flag=args.flag
    #venv=gym.make(env_id)
    #env=make_vec_env(venv)
    env= SoloMPC(exp_name,flag) 
    #venv =  DummyVecEnv([SoloMPC])
    #venv=gym.make(env_id)
    #print(venv.envs[0])
    #env=venv.env
    #print(env.pin_robot)
    a2c_student = A2C('MlpPolicy', env, verbose=1)
    if flag==0:
        train_bc(env,a2c_student,exp_name)
    else:
        visualise(env,10,True,exp_name)
    #collect_dataset(venv,exp_name)
    #train_dagger(env,a2c_student,exp_name)
    #train_gail(venv)
   

    

