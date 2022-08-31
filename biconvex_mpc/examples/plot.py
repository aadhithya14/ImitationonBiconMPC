import matplotlib.pyplot as plt
import json
import os
import numpy as np
#from imitation.algorithms.bc import reconstruct_policy
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from imitation.algorithms.bc import reconstruct_policy

def plot_reward(loss,neglogploss,ent_loss,l2_loss,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(4,1)

    #plt.plot(loss)
    ax[0].plot(loss)
    ax[0].set_title('total_loss')
    ax[1].plot(neglogploss,label='negativelogploss')
    ax[1].set_title('negativelogploss')
    ax[2].plot(ent_loss,label='entropy_loss')
    ax[2].set_title('entropyloss')
    ax[3].plot(l2_loss)
    ax[3].set_title('l2_loss')
   
    plt.ylabel("losses")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"mean_reward.png")

def plot_reward(loss,neglogploss,ent_loss,l2_loss,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(4,1)

    #plt.plot(loss)
    ax[0].plot(loss)
    ax[0].set_title('total_loss')
    ax[1].plot(neglogploss,label='negativelogploss')
    ax[1].set_title('negativelogploss')
    ax[2].plot(ent_loss,label='entropy_loss')
    ax[2].set_title('entropyloss')
   
   
    plt.ylabel("losses")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"mean_reward.png")



def plot_mean_loss(loss,neglogploss,ent_loss,l2_loss,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(4,1)
    mean_length=int(len(loss)/50)
    mean_loss=[]
    mean_loss_neglogp=[]
    mean_loss_ent_loss=[]
    for i in range(50):
        loss_mean=np.mean(loss[mean_length*i:mean_length*(i+1)])
        mean_loss.append(loss_mean)
        loss_mean_neglogp=np.mean(neglogploss[mean_length*i:mean_length*(i+1)])
        mean_loss_neglogp.append(loss_mean_neglogp)
        loss_mean_neglogp=np.mean(ent_loss[mean_length*i:mean_length*(i+1)])
        mean_loss_neglogp.append(loss_mean_neglogp)



    #plt.plot(loss)
    ax[0].plot(mean_loss)
    ax[0].set_title('total_loss')
    ax[1].plot(neglogploss,label='negativelogploss')
    ax[1].set_title('negativelogploss')
    ax[2].plot(ent_loss,label='entropy_loss')
    ax[2].set_title('entropyloss')
    ax[3].plot(l2_loss)
    ax[3].set_title('l2_loss')
   
    plt.ylabel("losses")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"mean_loss.png")

def plot_loss(loss,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    

    plt.plot(loss)
   
    plt.ylabel("losses")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"loss.png")

def plot_qdes(qdes,qdes_policy,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(12,1)
    qdes=np.transpose(qdes)
    qdes_policy=np.transpose(qdes_policy)
    for i in range(12):
        ax[i].plot(qdes[i],scaley=True)
        ax[i].plot(qdes_policy[i])
    plt.ylabel("qdes")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    #plt.show()
    plt.savefig(exp_name+"qdes.png")

def plot_torque(torque,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(12,1)
    torque=np.transpose(torque)
    for i in range(12):
        ax[i].plot(torque[i],scaley=True)
    plt.ylabel("torque")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    #plt.show()
    plt.savefig(exp_name+"torque.png")

def plot_obs(qdes,qdes_policy,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(12,1)
    #obs=np.transpose(obs)
    #obs=obs[0:2500]
    qdes=np.transpose(qdes)
    qdes_policy=np.transpose(qdes_policy)
    for i in range(12):
        ax[i].plot(qdes[i],scaley=True,color='r',label='mpc_output')
        ax[i].legend(loc='upper left')
        ax[i].plot(qdes_policy[i],scaley=True,color='b',label='policy_output')
        ax[i].legend(loc='upper left')
    plt.ylabel("qdes")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    #plt.show()
    plt.savefig(exp_name+"trajectory.png")



def plot_logs(log,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax= plt.subplots(4,3)
    base_pos=log['base_pos']
    feet_pos=log['feet_pos']
    feet_pos=np.transpose(feet_pos)
    base_pos=np.transpose(base_pos)
    print(base_pos)
    print(feet_pos)
    diff_pos=log['diff_pos']
    diff_pos=np.transpose(diff_pos)
    print(diff_pos)
    activation=log['activation']
    """ax[0][0:4].plot(base_pos,label='base_position')
    ax[0].set_title('base_position')"""
    cnt=0
    for j in range(4):
        for i in range(3):

            ax[j][i].plot(feet_pos[0][0][cnt],label='feet_position')
            #ax[1][i].set_title('feet_position')
            #ax[0][i].plot(base_pos[i],label='base_position')
            ax[j][i].plot(diff_pos[cnt],label='diff_position')
            cnt+=1
            #ax[2][i].plot(diff_pos[i],label='difference')
            #ax[2][i].set_title('difference')
    """ax[3][0:4].plot(activation,label='activation')
    ax[3].set_title('activation')"""
   
    
    #plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"logs.png")

def plot_actions(action,action_des,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    

    plt.plot(action,'b',label='qdes_actual')
    plt.plot(action_des,'r',label='qdes')
   
    plt.ylabel("actions")
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"actions.png")

def plot_actions_true(exp_name):
    
    #plt.rcParams["figure.figsize"] = (20,20)
    
    log=json.load(open(exp_name+"obs_new.json"))
    mylog=Log(exp_name+"values_new.json")
    policy=reconstruct_policy(exp_name+"/policy")
    for n in range(2500):
        #n=np.random.randint(0,150000)
        obs=np.array(log['obs'][2500+n])
        action_des=np.array(log['acts'][2500+n])
        print(action_des)
        action_policy,_=policy.predict(obs)
        print(action_policy)
        mylog.add('qdes',action_des.tolist())
        mylog.add('qdes_policy',action_policy.tolist())
    mylog.save()
    #plt.plot(action,'b',label='qdes_actual')
    #plt.plot(action_des,'r',label='qdes')
   
    #plt.ylabel("actions")
    #plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    #plt.savefig(exp_name+"actions.png")


def plot_horizon(horizon,exp_name):
    plt.rcParams["figure.figsize"] = (20,20)
    
    
    plt.bar([1,2,3,4],horizon,width=0.2)
   
    plt.ylabel("failures")
    plt.xlabel('plan_frequency')
    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=None, rect=None)
    plt.savefig(exp_name+"horizon.png")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--f', help='')
    parser.add_argument('--exp', help='')
    parser.add_argument('--name',help='experiment name')
    parser.add_argument('--train', help='')
    args = parser.parse_args()

    folder = args.f
    exp_name=args.name
    exp_num = 0
    exp_folder = folder

    reward_list = []



    
    #log=json.load(open(exp_folder+"losses.json"))
    #while (os.path.isdir(exp_folder)):
    #log=json.load(open(exp_folder+"t.json"))
    #log=json.load(open(exp_folder+"values_new.json"))
    #log=json.load(open(exp_folder+"actions.json"))
        #ep_log=json.load(open(exp_folder+'logdir.monitor.json'))
    #actual_qdes=log['actual_qdes']
    #qdes=log['qdes_command']
    #plot_actions_true(exp_name)
    #torque=log['torque']
    #qdes=log['qdes']
    #qdes_policy=log['qdes_policy']
    #obs=log['obs']
    qdes=log['qdes']
    qdes_policy=log['qdes_policy']
    #plot_actions(qdes,actual_qdes,exp_name)
    #neglogploss=log["neglogp"]
    #ent_loss=log["ent_loss"]
    #loss=log["loss"]
    #loss_list.append(reward)
    #exp_num += 1
    #exp_folder = folder + str(exp_num).zfill(3) + '/'

    #plot_loss(loss,exp_name)
    #plot_torque(obs,exp_name)
    #plot_obs(qdes,qdes_policy,exp_name)
    #plot_qdes(actions,actions_policy,exp_name)
    #horizon=[35,9,8,9]
    #plot_horizon(horizon,exp_name)