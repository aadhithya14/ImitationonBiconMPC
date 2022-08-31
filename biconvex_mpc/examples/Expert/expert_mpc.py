import yaml
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
from imitation.algorithms.bc import reconstruct_policy
from motions.cyclic.solo12_trot import trot
from controllers.robot_id_controller import InverseDynamicsController
import json
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
import numpy as np
import pybullet
import time


# Expert to sample expert trajectories

class Expert():
    def __init__(self,env,exp_name):
        self.env = env
        self.exp_name = exp_name
        self.exp_config = yaml.load(open(exp_name + 'conf.yaml'))
        self.w_des = 0.0

        self.plan_freq = 0.05 # sec
        self.update_time = 0.0 # sec (time of lag)

        self.sim_t = 0.0
        self.sim_dt = .001
        self.index = 0
        self.pln_ctr = 0
        #state=np.zeros(2)
        self.obs=self.env.reset()
        #x0=env.reset()
        #print(x0.shape[0])
        #x0=np.concatenate((state,obs))
        #x0=env.reset()
        self.x0=self.obs[0:37]
        ## Motion
        
        self.gait_params = trot
        self.lag = int(self.update_time/self.sim_dt)
        self.gg = SoloMpcGaitGen(self.env.pin_robot, self.env.urdf_path, self.x0, self.plan_freq,self.x0[0:19], None)

        self.gg.update_gait_params(self.gait_params, self.sim_t)

        #robot = PyBulletEnv(Solo12Robot, q0, v0)
        self.robot_id_ctrl = InverseDynamicsController(self.env.pin_robot, self.env.f_arr)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)

        self.plot_time = 0 #Time to start plotting
        self.trajectory_list=[]
        self.action_space="pd"
        self.kp=2.0
        self.kd=0.05
        self.solve_times = []
        self.failure=0
        self.done=False
        #self.policy=reconstruct_policy(exp_name+"/120_policy")
        self.log=Log(exp_name+"obs_new.json")
        #self.log=json.load(open(self.exp_name+"obs_new.json"))
        #self.mylog=Log(self.exp_name+"values.json")
        self.timesteps=self.exp_config['timesteps']
        self.steps=self.exp_config['steps']
        #self.collect=collect
    #trajector
    def unscale_action(scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = -3.145,3.145
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


    def rollout(self):
        for i in range(self.timesteps):
        #state=np.zeros(2)
            self.obs=self.env.reset()
            #x0=np.concatenate((state,obs))

            #x0=env.reset()
            self.v_des = np.array([self.obs[37],0.0,0.0])
            self.x0=self.obs[0:37]
        
            self.w_des = 0.0
            self.trajectories=[]
            for i in range(500):
                self.env.qdes_act(self.x0[7:19])
                pybullet.stepSimulation()
            
                
            #log.save()
            if self.done==True:
                print("***")
                self.failure+=1
            self.done=False
            self.plan_freq = 0.05 # sec
            self.update_time = 0.0 # sec (time of lag)
            
            self.q,self.v=self.env.robot.get_state()
            self.x=np.concatenate((np.array(self.q),np.array(self.v),self.v_des,np.array([self.w_des])))
            #self.x=self.log['obs'][0]
            self.x=np.array(self.x[0:37])

            self.sim_t = 0.0
            self.sim_dt = .001
            self.index = 0
            self.pln_ctr = 0
            self.lag = int(self.update_time/self.sim_dt)
            #x0=np.array(log['obs'][0][0:37])
            self.gg = SoloMpcGaitGen(self.env.pin_robot, self.env.urdf_path, self.x, self.plan_freq,self.x[0:19], None)

            self.gg.update_gait_params(self.gait_params, self.sim_t)
            self.phase=np.zeros(4)
            #self.obs=self.log['obs'][0]
            
            for o in range(int(self.steps*(self.plan_freq/self.sim_dt))):
                    # this bit has to be put in shared memory
                #if not done:
                    q, v = self.env.robot.get_state()
                    for j in range(4):
                        self.phase[j] = self.gg.gait_planner.get_percent_in_phase(np.round(self.sim_t,3), j)
                    print("phase", np.round(self.phase,4), self.sim_t)
                   # assert False
                    self.obs=np.concatenate((np.array(q),np.array(v),self.v_des,np.array([self.w_des]),self.phase))
                    
                    #obs=env.reset()
                    #print(obs)
                        #obs=obs.tolist()
                        #trajectories["obs"]=obs
                        
                        # if o == int(50*(plan_freq/sim_dt)):
                        #     gait_params = trot
                        #     gg.update_gait_params(gait_params, sim_t)
                        #     robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

                    if self.pln_ctr == 0:
                        self.contact_configuration = self.env.robot.get_current_contacts()
                            
                        pr_st = time.time()
                        self.xs_plan, self.us_plan, self.f_plan = self.gg.optimize(np.array(self.obs[0:19]), np.array(self.obs[19:37]), np.round(self.sim_t,3), np.array(self.obs[37:40]), np.array(self.obs[40]))
                            # Plot if necessary
                            # if sim_t >= plot_time:
                                # gg.plot_plan(q, v)
                                # gg.save_plan("trot")

                        pr_et = time.time()
                        self.solve_times.append(pr_et - pr_st)

                        # first loop assume that trajectory is planned
                    if o < int(self.plan_freq/self.sim_dt) - 1:
                        self.xs = self.xs_plan
                        self.us = self.us_plan
                        self.f = self.f_plan

                        # second loop onwards lag is taken into account
                    elif self.pln_ctr == self.lag and o > int(self.plan_freq/self.sim_dt)-1:
                            # Not the correct logic
                            # lag = int((1/sim_dt)*(pr_et - pr_st))
                        self.lag = 0
                        self.xs = self.xs_plan[self.lag:]
                        self.us = self.us_plan[self.lag:]
                        self.f = self.f_plan[self.lag:]
                        self.index = 0

                    tau = self.robot_id_ctrl.id_joint_torques(np.array(self.obs[0:19]), np.array(self.obs[19:37]), self.xs[self.index][:self.env.pin_robot.model.nq].copy(), self.xs[self.index][self.env.pin_robot.model.nq:].copy()\
                                                    , self.us[self.index], self.f[self.index], self.contact_configuration)

                    if self.action_space=="torque":
                        next_obs,reward,done,info =self.env.step(tau)
                    elif self.action_space=="increased_frequency_pd":
                        action=self.xs[index+1][7:self.env.pin_robot.model.nq].copy()
                        next_obs,reward,done,info =self.env.step(action)
                    else:
                        self.qdes=self.env.find_qdes(tau,self.kp,self.kd)
                        self.new_obs,self.reward,self.done,self.info =self.env.step(self.qdes)
                        low, high = -3.1415,3.1415
                        #print(low)
                        self.qdes= 2.0 * ((self.qdes - low) / (high - low)) -1.0
                        print(self.qdes)
                        #self.mylog.add('qdes',self.qdes.tolist())
                        #or i in range(8):
                        
                        #low, high = self.env.action_space.low, self.env.action_space.high
                        #print(low)
                        #self.qdes= 2.0 * ((self.qdes - low) / (high - low)) - 1.0
                        #print("###")

                    self.sim_t += self.sim_dt
                    self.pln_ctr = int((self.pln_ctr + 1)%(self.plan_freq/self.sim_dt))
                    self.index += 1

                    if self.done:
                        print("###")
                        #pybullet.stopStateLogging(logging_id)
                        break

                    self.trajectory={'obs':self.obs,'acts':self.qdes,'torque':tau}
                    self.trajectories.append(self.trajectory)
                    #print(done)
            
            if not self.done:
                self.trajectory_list.extend(self.trajectories)
                for traj in self.trajectories:
                    #print(type(traj['obs']))
                    self.log.add('obs',traj['obs'].tolist())
                    self.log.add('acts',traj['acts'].tolist())
                    self.log.add('torque', traj['torque'].tolist())
            

        self.log.save()
        print(self.failure)
        return self.trajectory_list