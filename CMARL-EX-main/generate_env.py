import pandapower as pp
import pandapower.networks as networks
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
import os

from envs.smart_grid.energy_models import Building, Weather

##generate net
# make a grid that fits the buildings generated
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

pv_buses = [item[-1] for item in conns]
pv_buses += [item[-2] for item in conns]

mapping = {18: 1, 25: 5, 22: 2}

net.line.drop(index=net.line[net.line.in_service == False].index, inplace=True)
net.bus_geodata.at[0, 'x'] = 0
net.bus_geodata.at[0, 'y'] = 0
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
                net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[mapping[c[i]], sw] + 0.2
                net.bus_geodata.at[c[i], st] = net.bus_geodata.at[mapping[c[i]], st]
        else:
            net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[c[i - 1], sw] + 0.2
            net.bus_geodata.at[c[i], st] = net.bus_geodata.at[c[i - 1], st]

net.ext_grid.at[0, 'vm_pu'] = 1.01

pp.create_shunt_as_capacitor(net, 14, 1.2, 0)
pp.create_shunt_as_capacitor(net, 24, 0.6, 0)
pp.create_shunt_as_capacitor(net, 30, 1.2, 0)


## add houses

with open("./config.yaml") as f:
    config_dict = yaml.safe_load(f)
# env config
env_config_dict = config_dict['environment']
climate_zone = env_config_dict['climate_zone']
data_path = Path("./envs/data/Climate_Zone_" + str(climate_zone))
weather_file = os.path.join(data_path, "weather_data.csv")
solar_file = os.path.join(data_path, "solar_generation_1kW.csv")
hourly_timesteps = env_config_dict['hourly_timesteps']
weather = Weather(weather_file, solar_file, hourly_timesteps)
buildings_states_actions_file = "./envs/data/buildings_state_action_space.json"
save_memory = True
houses_per_node = env_config_dict['houses_per_node']

n_agents = env_config_dict['houses_per_node']*32

n = houses_per_node
m = n
houses = []
b = 0
scaling_number = 6 / n

# find nodes in the network with residential voltage levels and load infrastructure
# get the node indexes by their assigned names
ext_grid_nodes = set(net.ext_grid['bus'])
res_voltage_nodes = set(net.bus['name'][net.bus['vn_kv'] == 12.66])
res_load_nodes = res_voltage_nodes - ext_grid_nodes

buildings = {}
for existing_node in list(res_load_nodes):
    # remove the existing arbitrary load
    net.load.drop(net.load[net.load.bus == existing_node].index, inplace=True)

    # add n houses at each of these nodes
    BuildingId = 0
    for i in range(m):
        BuildingId += 1
        with open(buildings_states_actions_file) as file:
            buildings_states_actions = json.load(file)
        building_ids = list(buildings_states_actions.keys())
        prob = np.ones(len(building_ids))
        prob[[1,4,5,6,7,8]] = 10
        prob = prob / sum(prob)
        uid = np.random.choice(building_ids, p=prob)
        bldg = Building(data_path, climate_zone, buildings_states_actions_file, hourly_timesteps,
                        uid, weather, BuildingId, save_memory=save_memory)
        bldg.assign_bus(existing_node)
        bldg.load_index = pp.create_load(net, bldg.bus, 0, name=bldg.buildingId,
                                         scaling=scaling_number)  # create a load at the existing bus
        if np.random.uniform() <= 2:  # equivalent to 100% PV penetration
            bldg.gen_index = pp.create_sgen(net, bldg.bus, 0, name=bldg.buildingId,
                                            scaling=scaling_number)  # create a generator at the existing bus
        else:
            bldg.gen_index = -1

        buildings[bldg.buildingId] = bldg

# mkdir and export net and agents
if not os.path.exists("./envs/data/" + str(n_agents) + "_agents"):
    os.mkdir("./envs/data/" + str(n_agents) + "_agents")
path = "./envs/data/" + str(n_agents) + "_agents/"
pp.to_pickle(net, path + "case33.p")
file = path + "agent_" + str(n_agents) + "_zone_" + str(env_config_dict['climate_zone']) + ".pickle"
with open(file, "wb") as f:
    pickle.dump(buildings, f)


