# CMARL-EX


## Collecting Demonstrations

To collect demonstrations, we use the RBC method to obtain trajectories. Then we collect demonstrations with different reward and then we can derive the ranking.


## Pre-Training

### The structure of the demonstrations files
Each demonstration file is a pickle file of a dict `{'traj':[traj_1, traj_2, ..., traj_N], 'reward':[reward_1, reward_2, ..., reward_N]}`



### Use partial trajectories 
`
cd CMARL-EX-main

python traintrex_sum.py --train_demo_files model_save/multi_model_1.pt model_save/multi_model_2.pt model_save/multi_model_3.pt model_save/multi_model_4.pt model_save/multi_model_5.pt model_save/multi_model_6.pt --test_demo_files model_save/multi_model_1.pt model_save/multi_model_3.pt model_save/multi_model_5.pt model_save/multi_model_7.pt
`

## Training

`
cd CMARL-EX-main

python safe_train_grid_ex.py 
`

## Testing
`
cd CMARL-EX-main

python safe_testing.py 
`

## Environment
This project constructs an energy management environment based on the CityLearn framework, with code modifications available here. (https://github.com/intelligent-environments-lab/CityLearn)
CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. 

## Important Note
This project provides the case study based solely on the IEEE 33-bus system. Data from the custom-built 86-bus system is not available for sharing.