#Gym Environment to be used by imitation learning algorithms
#Author : Aadhithya Iyer
from logging import info
import gym,gym.envs
from gym.spaces import Box
import pybullet
import pinocchio as pin
import tempfile
from envs.pybullet_env import PyBulletEnv
from bullet_utils.env import BulletEnvWithGround
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
import matplotlib.pyplot as plt
import numpy as np
from kinematic_controllers.position_gain_controller import PositionGainController
from rewards.Velocityreward import Reward
from termination.termination_all import Termination
import yaml
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from utils.my_math import scale
import random

class SoloMPC(gym.Env):
    def __init__(self,exp_name,flag,plot=0):
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.urdf_path = Solo12Config.urdf_path
        self.n_eff = 4
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.des=0.2
        self.wdes=0.0
        self.x0 = np.concatenate((self.q0, pin.utils.zero(self.pin_robot.model.nv)))
        self.f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.robot = PyBulletEnv(Solo12Robot, self.q0, self.v0)
        self.obs_dim=45
        self.action_dim=12
        self.ndof=12
        self.high= np.inf * np.ones([self.obs_dim])
        self.observation_space=Box(-self.high, self.high)
        self.high_action=np.ones([self.action_dim])
        self.low_action=np.zeros([self.action_dim])
        self.action_space=Box(-self.high_action, self.high_action)
        self.exp_name=exp_name
        self.exp_config=yaml.load(open(exp_name + 'conf.yaml'))
        self.plot=plot
        #np.random.choice([-0.1,-0.05, 0.0,0.05,0.1])
        self.num_envs=1
        self.flag=flag 
        self.cnt=0     
        self.reward_list=[] 
        self.termination=Termination(self.robot) 
        self.max_torque=2.5 
        self.joint_limits=[[-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415], [-3.1415, 3.1415] ]
        self.plotlog=Log(exp_name+"logs.json")
        self.failure=0
        self.history=[]

    def reset(self):
       
        self.done=False
        self.q0 = np.array(Solo12Config.initial_configuration)
        self.loop=0
        self.q0[0:2] =0.0
        self.v0 = pin.utils.zero(self.pin_robot.model.nv)
        self.q0[7] = np.random.uniform(-0.2,0.2)
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
        
        self.robot.robot.reset_state(self.q0, self.v0)
        if self.exp_config['state_conf']=='x_y_random':
            self.q0[0] = np.random.uniform(-0.2,0.2)
            self.q0[1] = np.random.uniform(-0.2,0.2)

        elif self.exp_config['state_conf']=='yaw_random':
            self.q0[0] = np.random.uniform(-0.2,0.2)
            self.q0[1] = np.random.uniform(-0.2,0.2)
            euler=list(pybullet.getEulerFromQuaternion([self.q0[3],self.q0[4],self.q0[5],self.q0[6]]))
            euler[2]+=np.random.uniform(-0.2,0.2)
            quaternion=pybullet.getQuaternionFromEuler(euler)
            self.q0[3:7]=quaternion

        elif self.exp_config['state_conf']=='contact_activation_without_x_and_y':
            self.activation=self.robot.get_current_contacts()
            print(self.activation)
        
        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y':
            print("//////")
            #print(feet_position[0])
            feet_position= []
            self.diff_pos=[]
            self.robot.robot.reset_state(self.q0, self.v0)
            self.base_pos=self.q0[0:3]
            for i in range(4):
                #print(self.robot.robot.end_eff_ids[i])
                print(pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0])
                self.feet_pos=pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0]
                feet_position.append(self.feet_pos)
                self.diff_pos.append(self.base_pos-self.feet_pos)
            print(self.diff_pos)
            #print(feet_position)

        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y_with_contacts':
            print("//////")
            #print(feet_position[0])
            feet_position= []
            self.diff_pos=[]
            self.robot.robot.reset_state(self.q0, self.v0)
            self.base_pos=self.q0[0:3]
            for i in range(4):
                #print(self.robot.robot.end_eff_ids[i])
                #print(pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0])
                self.feet_pos=pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0]
                feet_position.append(self.feet_pos)
                self.diff_pos.append((self.base_pos-self.feet_pos).tolist())
            print(self.base_pos)
            print(feet_position)
            print(self.diff_pos)
            self.activation=self.robot.get_current_contacts().tolist()
            print(self.activation)
            self.feet_pos=list(self.feet_pos)
            self.base_pos=self.base_pos.tolist()
            if self.plot==1:
                self.plotlog.add('feet_pos',feet_position)
                self.plotlog.add('base_pos',self.base_pos)
                self.plotlog.add('diff_pos',self.diff_pos)
                self.plotlog.add('activation',self.activation)
                print("$$$$$$$$$$$$$$$$$$$$")
            print(type(self.activation))
            print(type(self.feet_pos))
            print(type(self.base_pos))
            print(type(self.diff_pos))

        elif self.exp_config['state_conf']=='sensor_space':
            self.robot.robot.compute_numerical_quantities(dt=0.001)
            linacc=self.robot.robot.get_base_imu_linacc()
            ang_vel=self.robot.robot.get_base_imu_angvel()
            q,v=self.robot.robot.get_state()
            encoder_positions=q[7:]
            encoder_velocities=q[6:]
            encoder_velocities+=np.random.normal(0, .1, encoder_velocities.shape)
            self.state=np.concatenate((linacc,ang_vel,encoder_positions,encoder_velocities))
        elif self.exp_config['state_conf']=='encoder_space':
            self.robot.robot.compute_numerical_quantities(dt=0.001)
            linacc=self.robot.robot.get_base_imu_linacc()
            ang_vel=self.robot.robot.get_base_imu_angvel()
            q,v=self.robot.robot.get_state()
            encoder_positions=q[7:]
            encoder_velocities=q[6:]
            encoder_velocities+=np.random.normal(0, .1, encoder_velocities.shape)
            self.state=np.concatenate((linacc,ang_vel,encoder_positions,encoder_velocities))
            self.history.append(self.state)
            
            self.history=[]
            
        
        self.des=-0.1#np.random.uniform(-0.2,0.2)
        #self.des=0.0
        self.wdes=0.0

        self.phase=[0, 0.8333, 0.8333, 0]
        
        self.termination.init_termination()
        for termination_object in self.termination.termination_dict.values():
            #print(termination_object)
            termination_object.reset()
        
        #self.des=np.random.choice([-0.1,-0.05])
        #self.plotlog.save()
        #print(position)
        

        
        self.cnt=0
        
        if self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y':
            self.x0 = np.concatenate((self.q0[2:], np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.diff_pos[0],self.diff_pos[1],self.diff_pos[2],self.diff_pos[3]))
        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y_with_contacts':
            self.x0 = np.concatenate((self.q0[2:], np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.diff_pos[0],self.diff_pos[1],self.diff_pos[2],self.diff_pos[3],self.activation))
        elif self.exp_config['state_conf']=='contact_activation_without_x_and_y':
            self.x0 = np.concatenate((self.q0[2:], np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.activation))
        elif self.exp_config['state_conf']=='no_x_and_y':
            self.x0 = np.concatenate(self.q0[2:], np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]))
        elif self.exp_config['state_conf']=='encoder_space':
            self.x0 = np.concatenate(self.q0, np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]))
        else:
            self.x0 = np.concatenate((self.q0, np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes])))
        

        #print(self.x0.shape[0])
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
        
        for i in range(4):
            self.phase[i]+=0.003333

        if self.flag!=0:
            self.done=False
            #print("**************")
            if self.cnt>=30000:
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
        
        """
        if self.cnt%100==0:
            pos=[[-1,0,0],[1,0,0],[0,0,0],[0,-1,0],[0,1,0]]
            position=random.choice(pos)
            print(position)
            pybullet.applyExternalForce(self.robot.robot.robotId,-1,[np.random.uniform(-2,2),np.random.uniform(-10,10),np.random.uniform(-5,5)],position,flags=pybullet.LINK_FRAME)
        """
        #print(self.phase)

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

        elif self.exp_config['state_conf']=='contact_activation_without_x_and_y':
            self.activation=self.robot.get_current_contacts()
            print(self.activation)    
        
        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y':
            print("//////")
            #print(feet_position[0])
            feet_position= []
            self.diff_pos=[]
            #self.robot.robot.reset_state(self.q, self.v0)
            self.base_pos=self.q[0:3]
            for i in range(4):
                #print(self.robot.robot.end_eff_ids[i])
                print(pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0])
                self.feet_pos=pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0]
                feet_position.append(self.feet_pos)
                self.diff_pos.append((self.base_pos-self.feet_pos).tolist())
            print(self.diff_pos)

        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y_with_contacts':
            print("//////")
            #print(feet_position[0])
            feet_position= []
            self.diff_pos=[]
            #self.robot.robot.reset_state(self.q0, self.v0)
            self.base_pos=self.q[0:3]
            for i in range(4):
                #print(self.robot.robot.end_eff_ids[i])
                print(pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0])
                self.feet_pos=pybullet.getLinkState(self.robot.robot.robotId, self.get_endeff_link_ids()[i])[0]
                feet_position.append(self.feet_pos)
                self.diff_pos.append((self.base_pos-self.feet_pos).tolist())
            print(self.base_pos)
            print(feet_position)
            print(self.diff_pos)
            self.activation=self.robot.get_current_contacts().tolist()
            print(self.activation)
            self.base_pos=self.base_pos.tolist()
            if self.plot==1:
                self.plotlog.add('feet_pos',feet_position)
                self.plotlog.add('base_pos',list(self.base_pos))
                self.plotlog.add('diff_pos',self.diff_pos)
                self.plotlog.add('activation',self.activation)
                #print("$$$$$$$$$$$$$$$$$$$$")
            print(type(self.activation))
            print(type(self.feet_pos))
            print(type(self.base_pos))
        

        elif self.exp_config['state_conf']=='encoder_space':
            self.robot.robot.compute_numerical_quantities(dt=0.001)
            linacc=self.robot.robot.get_base_imu_linacc()
            ang_vel=self.robot.robot.get_base_imu_angvel()
            q,v=self.robot.robot.get_state()
            encoder_positions=q[7:]
            encoder_velocities=q[6:]
            #encoder_velocities+=np.random.normal(0, .1, encoder_velocities.shape)
            self.state=np.concatenate((linacc,ang_vel,encoder_positions,encoder_velocities))
            #print(self.state.shape[0])
            hidden = (torch.randn(1, 1,4*self.state.shape[0]),
            torch.randn(1, 1, self.state.shape[0]))
            
            
            if self.cnt%5==0:
                
                self.loop+=1
                self.history.append(self.state)
                
                #self.history=[]
            
            lstm = torch.nn.LSTM(self.state.shape[0], self.state.shape[0])
            
            
            if self.cnt>=20:
                if self.loop==4:
                    #for i in inputs:
                    self.history=torch.Tensor(self.history)
                    self.history.view(len(self.history), 1, -1)
                    self.out, self.hidden = lstm(self.history, hidden)
                    self.state=self.hidden
                    self.history=[]
                    self.loop=0
                        
        #
        if self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y':
            self.x = np.concatenate((self.q[2:], np.array(self.v),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.diff_pos[0],self.diff_pos[1],self.diff_pos[2],self.diff_pos[3]))
        elif self.exp_config['state_conf']=='pos_base_and_pos_diff_without_x_and_y_with_contacts':
            self.x = np.concatenate((self.q[2:], np.array(self.v0),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.diff_pos[0],self.diff_pos[1],self.diff_pos[2],self.diff_pos[3],self.activation))
        elif self.exp_config['state_conf']=='contact_activation_without_x_and_y':
            self.x = np.concatenate((self.q[2:], np.array(self.v),np.array([self.des,0.0,0.0]),np.array([self.wdes]),self.activation))
        elif self.exp_config['state_conf']=='no_x_and_y':
            self.x = np.concatenate(self.q[2:], np.array(self.v),np.array([self.des,0.0,0.0]),np.array([self.wdes]))
        elif self.exp_config['state_conf']=='encoder_space':
            self.x = self.state
        else:
             self.x = np.concatenate((self.q, np.array(self.v),np.array([self.des,0.0,0.0]),np.array([self.wdes])))
        #print(self.x.shape[0])

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

        q, v    = self.robot.get_state()
        #self.qdes_for_ttr = self.qdes
        #self.qdotdes_for_ttr = self.qdotdes
        v=v[6:]
        joint_type = ['circular'] * 12
        q_diff     = PositionGainController.pos_diff(q[7:], qdes, joint_type)

        torque = kp * q_diff - kd * v

        self.torque_control(torque)

    def torque_control(self, des_torque, no_clipping=True):
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

    def get_endeff_link_ids(self):  
        return [3, 7, 11, 15]   