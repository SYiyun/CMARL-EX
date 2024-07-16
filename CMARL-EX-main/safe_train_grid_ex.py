import sys
sys.path.append(".")
from sau_smartgrid_ex import SauteEnv
from pathlib import Path
from agents.MADDPG_C import MADDPG
import yaml
import test_models as model
import argparse
import pickle
import numpy as np
import torch as th
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
device = th.device("cuda:7" if th.cuda.is_available() else "cpu")
import csv
from ptflops import get_model_complexity_info
from torchsummary import summary
import time
parser = argparse.ArgumentParser()
argv = parser.parse_args()



reward_record = []
with open("./config.yaml") as f:
    config_dict = yaml.safe_load(f)
# env config
env_config_dict = config_dict['environment']
data_path = Path("./envs/data/Climate_Zone_" + str(env_config_dict['climate_zone']))
buildings_states_actions = './envs/data/buildings_state_action_space.json'

n_agents = env_config_dict['houses_per_node'] * 32
path = "./envs/data/" + str(n_agents) + "_agents/"

grid_config_dict = {
    "model_name": str(env_config_dict['houses_per_node'] * 32) + "agents",
    "data_path": data_path,
    "climate_zone": env_config_dict['climate_zone'],
    "buildings_states_actions_file": buildings_states_actions,
    "hourly_timesteps": 4,
    "max_num_houses": None,
    "houses_per_node": env_config_dict['houses_per_node'],
    "net_path": path + "case33.p",
    "agent_path": path + "agent_" + str(n_agents) + "_zone_" + str(env_config_dict['climate_zone']) + ".pickle"
}
env = SauteEnv(**grid_config_dict)


n_agents = env.get_num_of_agents()
print("agents_num:",n_agents)
n_states = env.get_obs_size()
n_actions = env.get_total_actions()
capacity = 1500
batch_size = 32

n_episode = 1000
#episodes_before_train = 4

max_steps = 32
exi_mini = 9

#maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,episodes_before_train)

reward_model = 'log/rewardnet_0.pth'
maddpg = th.load('./maddpg_model/pre_training_model.pt')
reward_net = model.RewardNet(n_states+ n_actions).float()
reward_net.load_state_dict(th.load(reward_model, map_location='cpu'))
env.set_reward_net(reward_net)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

print('---------------------------------start train-------------------------------')
for i_episode in range(n_episode):
    flag = 0
    stat = {}
    print('i_episode：',i_episode)
    obs_s, obs = env.reset(flag)
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    total_reward_all = 0.0
    total_realreward = 0.0
    total_realreward_all = 0.0
    rr = np.zeros((n_agents,))
    trajlist  = []
    rewardlist = []

    for exi in range(exi_mini):
        stat = {}
        obs = env.get_obs()
        env.get_state()
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        total_realreward = 0.0
        rr = np.zeros((n_agents,))
        trajlist = []
        rewardlist = []

        target_i = 0
        for t in range(max_steps):

            obs = obs.type(FloatTensor)
            action = maddpg.select_action(obs).data.cpu()

            s_, r, d, infor = env.step(action.numpy())

            vm_reward = env._step_get_real_reward()
            total_reward += np.array(r).sum()
            total_realreward += np.array(vm_reward).sum()
            r = th.FloatTensor(r).type(FloatTensor)
            reward = r
            obs_ = env.get_obs()


            real_reward = th.FloatTensor(vm_reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None


            rr += reward.cpu().numpy()
            maddpg.memory.push(obs.data, action, next_obs, reward)
            obs = next_obs

            c_loss, a_loss = maddpg.update_policy(target_i)
            target_i +=1
            info = infor
            for k, v in info.items():
                if k in stat.keys():
                    stat[k].append(v)

                else:
                    stat[k] = []
                    stat[k].append(v)
        env.reset2()
        maddpg.episode_done += 1
        print('Episode: %d, reward = %f, real_reward = %f' % (i_episode, total_reward,total_realreward))
        total_reward_all +=total_reward
        total_realreward_all += total_realreward
        for k, v in stat.items():
            v = np.array(v)
            print('The {} of our method is:{}'.format(k, v.sum()))
            with open('./paper_log/info.csv', 'a', encoding='utf-8') as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow([np.array(v.sum())])
        with open('./paper_log/loss.csv', 'a', encoding='utf-8') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([i_episode, total_reward, total_realreward])
        reward_record.append(total_reward)
    with open('./paper_log/loss_all.csv', 'a', encoding='utf-8') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([i_episode, total_reward_all,total_realreward_all])
    if i_episode % 50 == 0:
        #th.save({f'Actor{i}': maddpg.actors[i] for i in range(n_agents)}, "./maddpg_model/Actor.tar")
        #th.save({f'Critic{i}': maddpg.critics[i] for i in range(n_agents)}, "./maddpg_model/Critic.tar")
        th.save(maddpg, './maddpg_model/train/training_model_{}.pt'.format(i_episode))
        print('Model saved!')
