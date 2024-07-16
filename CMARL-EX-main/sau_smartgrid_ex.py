import numpy as np
import json
import torch
from numpy.linalg import matrix_power
from multiagentenv import MultiAgentEnv
import pandapower as pp
import os
from gym.utils import seeding
from pandapower import runpp
import pandapower.networks as networks
import pickle
from collections import Counter
from envs.smart_grid.energy_models import Building, Weather
import warnings
from typing import Union
Array = Union[torch.Tensor, np.ndarray]
warnings.filterwarnings("ignore")




class SafeGridEnv(MultiAgentEnv):
    """Safe Environment."""
    def __init__(self, model_name, data_path, climate_zone, buildings_states_actions_file, hourly_timesteps,
                 houses_per_node=6, cluster_adjacent_bus_num=6,
                 save_memory=True, building_ids=None, nclusters=2, randomseed=2, max_num_houses=None, percent_rl=1,
                 net_path="./", agent_path="./"):

        self.model_name = model_name
        self.max_num_houses = max_num_houses
        self.percent_rl = percent_rl

        self.cluster_adjacent_bus_num = cluster_adjacent_bus_num

        self.exipode = 0
        self.data_path = data_path
        self.climate_zone = climate_zone
        self.weather_file = os.path.join(self.data_path, "weather_data.csv")
        self.solar_file = os.path.join(self.data_path, "solar_generation_1kW.csv")
        self.weather = Weather(self.weather_file, self.solar_file, hourly_timesteps)
        self.buildings_states_actions_file = buildings_states_actions_file
        self.hourly_timesteps = hourly_timesteps
        self.save_memory = save_memory
        self.building_ids = building_ids

        self.net = pp.from_pickle(net_path)
        with open(agent_path, "rb") as f:
            self.buildings = pickle.load(f)

        self.agents = list(self.buildings.keys())
        self.possible_agents = self.agents[:]
        self.rl_agents = self._set_rl_agents()

        self.clusters = self._get_bus_clusters()

        self.observation_spaces = {k: v.observation_space for k, v in self.buildings.items()}
        self.action_spaces = {k: v.action_space for k, v in self.buildings.items()}

        self.metadata = {'render.modes': [], 'name': "gridlearn"}


        self.voltage_data = []
        self.load_data = []
        self.gen_data = []
        self.reward_data = []
        self.vm_reward_data = []
        self.all_rewards = []

        self.aspace, self.ospace = self._get_spaces(self.agents)
        self.single_agent_obs_size = self.ospace[self.agents[0]].shape[0]
        self.obs_size = self._get_partial_obs_max_len()
        self.state_size = self.single_agent_obs_size * len(self.agents)

        self.v_upper = 1.05
        self.v_lower = 0.95

        self.n_agents = len(self.agents)

        self.episode_limit = 20

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) :
        """Computes a linear safety cost between the current position
        (if its near the unsafe area, aka in the hazard region)
        and the centre of the unsafe region."""
        costs = self._cost()
        return costs

    def printts(self, reset_logs=True):
        return self._printts()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "action_space": self.buildings[self.agents[0]].action_space,
                    "agents_name": self.agents,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def set_reward_net(self, reward_net):
        self.reward_net = reward_net

    def _printts(self, reset_logs=True):
        print("self.ts", self.ts)

    def _reset(self, flag=0, reset_logs=True):
        rand_act = {k: v.sample() for k, v in self.aspace.items()}
        rand_act_array = np.array(list(rand_act.values()))
        self.steps = 0
        self.ts = self._select_timestamp(self.exipode)
        year_ts = self.ts % (8760*self.hourly_timesteps)
        self.net.shunt.at[0,'q_mvar'] = -1.2
        self.net.shunt.at[1,'q_mvar'] = -0.01
        self.net.shunt.at[2,'q_mvar'] = -0.01

        count = 0
        for agent in self.agents:
            self.buildings[agent].step(rand_act_array[count])
            count += 1

        self.ts += 1
        self.steps += 1

        # update the grid based on updated buildings
        self._update_grid()

        # run the grid power flow
        try:
            runpp(self.net, enforce_q_lims=True)
        except:
            pp.diagnostic(self.net)
            quit()
            print("QUITTING!!!!")

        for agent in self.agents:
            self.buildings[agent].reset_timestep(self.net, reset_logs)

        return self.get_obs(), self.get_state()




    def get_state(self):
        state = []
        for k in self.agents:
            state.append(self.buildings[k].get_state(self.net))
        return state

    def get_obs(self):
        all_state_dict = {k: np.array(self.buildings[k].get_state(self.net)) for k in self.agents}
        pad_obs_list = []
        for agent in self.rl_agents:
            agent_obs_array = np.concatenate([all_state_dict[neighbor] for neighbor in self.clusters[agent]])
            pad_obs_list.append(np.concatenate([agent_obs_array, np.zeros(self.obs_size-agent_obs_array.shape[0])]))
        return np.array(pad_obs_list)

    def get_obs_agent(self, agent_id):
        """return observation for agent_id
        """
        agents_obs = self.get_obs()
        return agents_obs[agent_id]

    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_num_of_agents(self):
        return len(self.agents)

    def get_total_actions(self):
        return self.aspace[self.agents[0]].shape[0]

    def get_avail_actions(self):
        return [4]

    def _select_timestamp(self,exipode):
        data_len = len(self.weather.data['t_out'])
        time_stamp = np.random.choice(data_len - self.episode_limit)
        time_stamp = 26208
        return time_stamp
    def _get_spaces(self, agents):
        # print([self.buildings[k].action_space for k in agents])
        actionspace = {k: self.buildings[k].action_space for k in agents}
        obsspace = {k: self.buildings[k].observation_space for k in agents}
        return actionspace, obsspace

    def _set_rl_agents(self):
        num_rl_agents = int(self.percent_rl * len(self.net.load.name))
        rl_agents = np.random.choice(self.net.load.name, num_rl_agents).tolist()
        return rl_agents

    def _make_grid(self):
        # make a grid that fits the buildings generated for CityLearn
        net = networks.case33bw()

        # clear the grid of old load values
        load_nodes = net.load['bus']
        res_voltage_nodes = net.bus['name'][net.bus['vn_kv'] == 12.66]
        res_load_nodes = set(load_nodes) & set(res_voltage_nodes)
        net.bus['min_vm_pu'] = 0.7
        net.bus['max_vm_pu'] = 1.3

        for node in res_load_nodes:
            # remove the existing arbitrary load
            net.load.drop(net.load[net.load.bus == node].index, inplace=True)

        conns = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21],
                [25, 26, 27, 28, 29, 30, 31, 32],
                [22, 23, 24]]

        self.pv_buses = [item[-1] for item in conns]
        self.pv_buses += [item[-2] for item in conns]

        mapping = {18:1, 25:5, 22:2}

        net.line.drop(index=net.line[net.line.in_service==False].index, inplace=True)
        net.bus_geodata.at[0,'x'] = 0
        net.bus_geodata.at[0,'y'] = 0
        sw = 'x'
        st = 'y'
        z = -1
        for c in conns:
            z += 1
            for i in range(len(c)):
                if i == 0:
                    if not c[i] == 0:
                        sw = 'y'
                        st = 'x'
                        net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[mapping[c[i]],sw] + 0.2
                        net.bus_geodata.at[c[i], st] = net.bus_geodata.at[mapping[c[i]],st]
                else:
                    net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[c[i-1], sw] + 0.2
                    net.bus_geodata.at[c[i], st] = net.bus_geodata.at[c[i-1], st]

        net.ext_grid.at[0,'vm_pu'] = 1.01

        pp.create_shunt_as_capacitor(net,14,1.2,0)
        pp.create_shunt_as_capacitor(net,24,0.6,0)
        pp.create_shunt_as_capacitor(net,30,1.2,0)
        return net



    def _update_grid(self):
        for agent, bldg in self.buildings.items():
            self.net.load.at[bldg.load_index, 'p_mw'] = 0.95 * bldg.current_gross_electricity_demand * 0.001
            self.net.load.at[bldg.load_index, 'sn_mva'] = bldg.current_gross_electricity_demand * 0.001

            if bldg.gen_index > -1:  # assume PV and battery are both behind the inverter
                self.net.sgen.at[bldg.gen_index, 'p_mw'] = -1 * bldg.current_gross_generation * np.cos(bldg.phi) * 0.001
                self.net.sgen.at[bldg.gen_index, 'q_mvar'] = bldg.current_gross_generation * np.sin(bldg.phi) * 0.001

    def _step_ex(self, actions):
        year_ts = self.ts % (8760 * self.hourly_timesteps)

        self.net.shunt.at[0, 'q_mvar'] = -1.2
        self.net.shunt.at[1, 'q_mvar'] = -0.01
        self.net.shunt.at[2, 'q_mvar'] = -0.01

        count = 0
        for agent in self.agents:
            self.buildings[agent].step(actions[count])
            count += 1

        self.ts += 1
        self.steps += 1

        # update the grid based on updated buildings
        self._update_grid()

        # run the grid power flow
        try:
            runpp(self.net, enforce_q_lims=True)
            # print(self.net.res_bus.p_mw.tolist())
        except:
            pp.diagnostic(self.net)
            quit()
            print("QUITTING!!!!")
        rl_agent_keys = self.agents
        #        obs = self.state(rl_agent_keys)
        self.voltage_data += [list(self.net.res_bus['vm_pu'])]

        self.load_data += [sum(list(self.net.load['p_mw']))]
        self.gen_data += [sum(list(self.net.sgen['p_mw']))]
        reward1 = self.get_rewardnet(actions)
        return self.get_obs(), reward1, self._get_done(), self._get_info()

    def _step_next(self, actions):

        self.net.shunt.at[0, 'q_mvar'] = -1.2
        self.net.shunt.at[1, 'q_mvar'] = -0.01
        self.net.shunt.at[2, 'q_mvar'] = -0.01

        reward = self.get_rewardnet(actions)
        return  reward

    def _step_get_real_reward(self):
        rewards, vm_rewards = self._get_reward()
        return rewards


    def get_rewardnet(self, action):
        observation_before = self.get_obs()
        observation_before = np.array(observation_before)

        #print(observation_before,np.clip(action, -1., 1.),np.concatenate([observation_before, np.clip(action, -1., 1.)], axis=1))
        net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, np.clip(action, -1., 1.)], axis=1))

        reward = net_reward  # - ctrl_cost
        return reward

    def _get_reward(self):
        rewards = {k: self.buildings[k].get_doreward(self.net) for k in self.agents}
        vm_rewards = {k: self.buildings[k].get_reward(self.net) for k in self.agents}
        self.reward_data += [sum(rewards.values())]
        self.vm_reward_data += [sum(vm_rewards.values())]
        return list(rewards.values()),list(vm_rewards.values())

    def _cost(self):
        rewards = {k: self.buildings[k].get_devs(self.net) for k in self.agents}
        return list(rewards.values())

    def _get_done(self):
        return self.buildings[self.agents[0]].time_step >= self.hourly_timesteps*8760

    def _get_info(self, info={}):
        demandloss = 0
        demandloss_power = 0
        car_all = 0
        v = self.net.res_bus['vm_pu'].sort_index().to_numpy()

        # percentage of voltage out of control
        percent_of_v_out_of_control = (np.sum(v < self.v_lower) + np.sum(v > self.v_upper)) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control

        # voltage violtation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean(np.abs(v - v_ref))
        info["average_voltage"] = np.mean(v)

        for k in self.agents:
            ca_power,car,de_power,de = self.buildings[k].get_carbonreward()
            demandloss += de
            car_all += car
        info["demand_loss"] = demandloss
        info["carbon_loss"] = car_all
        # line loss
        line_loss = np.sum(self.net.res_line["pl_mw"])
        avg_line_loss = np.mean(self.net.res_line["pl_mw"])
        info["total_line_loss"] = line_loss

        # reactive power (q) loss
        q = self.net.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss

        return info


    def _get_bus_clusters(self):
        # calc temp matrix that show adjacent bus
        G = np.eye(len(self.net.bus))
        id1 = self.net.line['from_bus'].tolist()
        id2 = self.net.line['to_bus'].tolist()
        G[id1, id2] = 1
        G[id2, id1] = 1
        temp = matrix_power(G, self.cluster_adjacent_bus_num)

        # clusters: BuildingsID --> adjacent bus BuildingsID
        clusters = dict()
        for agent in self.net.load['name'].tolist():
            agent_bus = self.net.load.loc[self.net.load['name']==agent, 'bus']
            adjacent_bus = np.where(temp[agent_bus].squeeze()>0)[0]
            clusters[agent] = self.net.load.loc[self.net.load['bus'].isin(adjacent_bus), 'name'].tolist()

        return clusters

    def _get_partial_obs_max_len(self):
        max_adjacent_agents = 0
        for adjacent_list in list(self.clusters.values()):
            if len(adjacent_list) > max_adjacent_agents:
                max_adjacent_agents = len(adjacent_list)

        return max_adjacent_agents * self.single_agent_obs_size



class SauteEnv(SafeGridEnv):
    def __init__(
            self,
            safety_budget: float = 1.0,
            saute_discount_factor: float = 0.99,
            max_ep_len: int = 32,
            min_rel_budget: float = 1.,  # minimum relative (with respect to safety_budget) budget
            max_rel_budget: float = 1.,  # maximum relative (with respect to safety_budget) budget
            test_rel_budget: float = 1.,  # test relative budget
            unsafe_reward: float = -200,
            use_reward_shaping: bool = True,  # ablation
            use_state_augmentation: bool = True,  # ablation
            **kwargs
    ):
        super().__init__(**kwargs)
        # wrapping the safe environment
        #self.wrap = SafeGridEnv(**kwargs)

        # dealing with safety budget variables
        assert safety_budget > 0, "Please specify a positive safety budget"
        assert saute_discount_factor > 0 and saute_discount_factor <= 1, "Please specify a discount factor in (0, 1]"
        assert max_ep_len > 0

        self.use_reward_shaping = use_reward_shaping
        self.use_state_augmentation = use_state_augmentation
        self.max_ep_len = max_ep_len

        self._saute_discount_factor = saute_discount_factor
        self._unsafe_reward = unsafe_reward
        self._safety_state = None
        self._safety_budget = None
        self.wrap = None
        self._mode = 'train'

        self.min_rel_budget = min_rel_budget
        self.max_rel_budget = max_rel_budget
        self.test_rel_budget = test_rel_budget
        if self.saute_discount_factor < 1:
            safety_budget = safety_budget * (1 - self.saute_discount_factor ** self.max_ep_len) / (
                        1 - self.saute_discount_factor) / np.float32(self.max_ep_len)
        self._safety_budget = np.float32(safety_budget)






    @property
    def safety_budget(self):
        return self._safety_budget

    @property
    def saute_discount_factor(self):
        return self._saute_discount_factor

    @property
    def unsafe_reward(self):
        return self._unsafe_reward

    def _augment_state(self, state:np.ndarray, safety_state:np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, safety_state]) if self.use_state_augmentation else state
        return augmented_state

    def safety_step(self, cost:np.ndarray) -> np.ndarray:
        """ Update the normalized safety state z' = (z - l / d) / gamma. """
        #self._safety_state = 1.00
        #self._safety_state = self.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)

        self._safety_state -= cost / self.safety_budget
        self._safety_state /= self.saute_discount_factor
        return self._safety_state

    def step(self, action):
        """ Step through the environment. """
        obs_now = self.get_obs()
        next_obs, reward, done, info = self._step_ex(action)

        info['cost'] = self._safety_cost_fn(obs_now, action, next_obs)
        next_safety_stateall = self.safety_step(info['cost'])
        reward_true = self._step_get_real_reward()
        info['true_reward'] = reward_true
        obs_list = []
        for ii in range(len(next_obs)):
            next_safety_state = next_safety_stateall[ii]
            reward[ii] = self.reshape_reward(reward[ii], next_safety_state)


            augmented_state = self._augment_state(next_obs[ii], next_safety_state)
            obs_list.append(augmented_state)
        return next_obs, reward, done, info



    def reset(self,flag = 0):
        """Resets the environment."""
        state,_ = self._reset(flag=flag)
        seed = 1234
        obs_list = []
        for ii in range(len(state)):
            self.np_random, seed = seeding.np_random(seed)
            if self._mode == "train":
                self._safety_state = self.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
            elif self._mode == "test" or self._mode == "deterministic":
                self._safety_state = self.test_rel_budget
            else:
                raise NotImplementedError("this error should not exist!")
            augmented_state = self._augment_state(state[ii], self._safety_state)
            obs_list.append(augmented_state)
        return obs_list,self.get_obs()

    def reset2(self) -> np.ndarray:
        state = self.get_obs()
        seed = 1234
        obs_list = []
        for ii in range(len(state)):
            self.np_random, seed = seeding.np_random(seed)
            if self._mode == "train":
                self._safety_state = self.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
            elif self._mode == "test" or self._mode == "deterministic":
                self._safety_state = self.test_rel_budget
            else:
                raise NotImplementedError("this error should not exist!")
            self._safety_state = self.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
            augmented_state = self._augment_state(state[ii], self._safety_state)
            obs_list.append(augmented_state)
        return obs_list

    def reshape_reward(self, reward: Array, next_safety_state: Array) -> Array:
        """ Reshaping the reward. """
        if self.use_reward_shaping:
            reward = reward * (next_safety_state > 0) + self.unsafe_reward * (next_safety_state <= 0)
        return reward
