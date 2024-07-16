import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

activation = nn.LeakyReLU

class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(RewardNet, self).__init__()
        self.input_dim = input_dim
        last_dim = self.input_dim
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(last_dim, hidden_dim))
            layer_list.append(activation())
            last_dim = hidden_dim
        layer_list.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)


    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = self.net(x)
            reward = torch.squeeze( reward)
        return reward

class RewardNets(nn.Module):
    def __init__(self, input_dim, net_num, hidden_dim=256, num_layers=2):
        super(RewardNets, self).__init__()
        self.reward_nets = nn.ModuleList([RewardNet(input_dim, hidden_dim, num_layers) for i in range(net_num)])
        self.net_num = net_num

    def forward(self, x_list):
        if type(x_list) == list and len(x_list) == self.net_num:
            return [self.reward_nets[i](x_list[i]) for i in range(self.net_num)]
        else:
            return [self.reward_nets[i](x_list) for i in range(self.net_num)]

    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = np.sum([self.reward_nets[i](x).item() for i in range(self.net_num)])
        return reward

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
