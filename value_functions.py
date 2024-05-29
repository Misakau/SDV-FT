import torch
import torch.nn as nn
from util import mlp

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

class TwinV(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, squeeze_output=True)
        self.v2 = mlp(dims, squeeze_output=True)

    def both(self, state):
        return self.v1(state), self.v2(state)

    def forward(self, state):
        return torch.min(*self.both(state))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)
    
class WeightNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + act_dim, 256, 128, 1]
        self.net = mlp(dims, output_activation=nn.ReLU, squeeze_output=True, set_last_bias=1.0)#, layer_norm=True)
        # non-negtive and not zero

    def forward(self, state, act):
        return self.net(torch.concat([state, act],dim=1)) 