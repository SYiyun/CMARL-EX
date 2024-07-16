

import sys
sys.path.append(".")
from sau_smartgrid_ex import SauteEnv
import gym
from pathlib import Path
from agents.MADDPG_C import MADDPG
import yaml
import argparse
import pickle
import numpy as np
import torch as th
import test_models as model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
numba = False
device = th.device("cuda:6" if th.cuda.is_available() else "cpu")
import csv


parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str, nargs="?", default="ippo")
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
    "model_name": str(env_config_dict['houses_per_node'] * 32) + "agents",  # default 6*32 = 192
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
print("states_num:",n_states,n_actions)
capacity = 1500
batch_size = 32



n_episode = 1
episodes_before_train = 3
max_steps = 32
exi_mini = 9

reward_model = 'log/rewardnet_0.pth'
reward_net = model.RewardNet(n_states+ n_actions).float()
reward_net.load_state_dict(th.load(reward_model, map_location='cpu'))
env.set_reward_net(reward_net)
maddpg = th.load('./maddpg_model/train/training_model.pt')
test_times =1
FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor



for i_episode in range(n_episode):

    stat = {}
    print('i_episode',i_episode)
    obs, obs_raw = env.reset()
    obs = np.stack(obs_raw)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    total_reward_all = 0.0
    total_realreward_all = 0.0
    rr = np.zeros((n_agents,))
    trajlist = []
    rewardlist = []
    for exi in range(exi_mini):
        stat = {}
        obs = env.get_obs()
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        total_realreward = 0.0
        rr = np.zeros((n_agents,))
        for t in range(max_steps):

            state_np = obs.numpy()
            obs = obs.type(FloatTensor)
            action = maddpg.select_action(obs).data.cpu()
            action_np = np.array(action)

            s_, r, d, infor = env.step(action.numpy())

            rewardnp = np.array(r)
            rewardlist.append([rewardnp])
            vm_reward = env._step_get_real_reward()
            total_realreward += np.array(vm_reward).sum()
            total_reward += np.array(r).sum()
            r = th.FloatTensor(r).type(FloatTensor)
            reward = r
            obs_ = env.get_obs()

            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None


            rr += reward.cpu().numpy()
            maddpg.memory.push(obs.data, action, next_obs, reward)
            obs = next_obs

            info = infor
            for k, v in info.items():
                if k in stat.keys():
                    stat[k].append(v)

                else:
                    stat[k] = []
                    stat[k].append(v)

        env.reset2()
        maddpg.episode_done += 1
        print('Episode: %d, reward = %f' % (i_episode, total_reward))
        total_reward_all +=total_reward
        total_realreward_all += total_realreward
        for k, v in stat.items():
            v = np.array(v)
            print('The {} of our method is:{}'.format(k, v.sum()))
            with open('./paper_log/test/info_ours_{}.csv'.format(k),'a', encoding='utf-8') as file_objX:
                writer = csv.writer(file_objX)
                writer.writerow(np.array(v))
            with open('./paper_log/test/info_all_ours.csv', 'a', encoding='utf-8') as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow([np.array(v.sum())])
        with open('./paper_log/test/loss_ours.csv', 'a', encoding='utf-8') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([i_episode, total_reward,total_realreward])
        reward_record.append(total_reward)
    with open('./paper_log/test/loss_all_ours.csv', 'a', encoding='utf-8') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([i_episode, total_reward_all,total_realreward_all])

