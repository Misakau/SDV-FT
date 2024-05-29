import csv
from datetime import datetime
from pathlib import Path
import random
import string
import sys
import os
import gym
from gym.core import Env

import numpy as np
import torch
import torch.nn as nn

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Normalizer(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.mean = np.zeros(env.observation_space.shape)
        self.std = np.ones(env.observation_space.shape)
        self.Nr = 1
        self.alpha = 0
        self.fine_tune = False

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        if self.fine_tune:
            self.std  = (1 - self.alpha) * self.std + self.alpha * np.abs(observation - self.mean)
            self.mean = (1 - self.alpha) * self.mean + self.alpha * observation

        observation = (observation - self.mean) / self.std
        reward *= self.Nr
        return observation, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        observation = (observation - self.mean) / self.std
        return observation
    
    def init(self, mean, std, Nr=1):
        assert self.mean.shape == mean.shape
        assert self.std.shape == std.shape
        self.mean = mean
        self.std = std
        self.Nr = Nr
    
    def set_fine_tune(self, flag):
        self.fine_tune = flag

def logistic_function(x):
    return .5 * (1 + np.tanh(.5 * x))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False, set_last_bias=None, layer_norm=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if layer_norm:
        layers.append(nn.LayerNorm(dims[-1]))
    if set_last_bias is not None:
        layers[-1].bias.data.fill_(set_last_bias)
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    
    """
    plt.figure()
    plt.hist(lengths)
    plt.savefig('hopper_.png')
    """
    print(f'total: {len(lengths)}, min length: {min(lengths)}, max length: {max(lengths)}, avg length: {sum(lengths)/len(lengths)}')
    return min(returns), max(returns)

def return_range_more(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])

    return returns, lengths


def extract_done_makers(dones):
    (ends, ) = np.where(dones)
    starts = np.concatenate(([0], ends[:-1] + 1))
    length = ends - starts + 1
    return starts, ends, length


def _sample_indces(dataset, batch_size):
    try: 
        dones = dataset["timeouts"].cpu().numpy()
    except:
        dones = dataset["terminals"].cpu().numpy()
    starts, ends, lengths = extract_done_makers(dones)
    # credit to Dibya Ghosh's GCSL codebase
    trajectory_indces = np.random.choice(len(starts), batch_size)
    proportional_indices_1 = np.random.rand(batch_size)
    proportional_indices_2 = np.random.rand(batch_size)
    # proportional_indices_2 = 1
    time_dinces_1 = np.floor(
        proportional_indices_1 * (lengths[trajectory_indces] - 1)
    ).astype(int)
    time_dinces_2 = np.floor(
        proportional_indices_2 * (lengths[trajectory_indces])
    ).astype(int)
    start_indices = starts[trajectory_indces] + np.minimum(
        time_dinces_1,
        time_dinces_2
    )
    goal_indices = starts[trajectory_indces] + np.maximum(
        time_dinces_1,
        time_dinces_2
    )

    return start_indices, goal_indices


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}

def rvs_sample_batch(dataset, batch_size):
    start_indices, goal_indices = _sample_indces(dataset, batch_size)
    dict = {}
    for k, v in dataset.items():
        if (k == "observations") or (k == "actions"):
            dict[k] = v[start_indices]
    dict["next_observations"] = dataset["observations"][goal_indices]
    dict["rewards"] = 0
    dict["terminals"] = 0
    return dict

def evaluate_sdv(eval_env, algo, mean=0.0, std=1.0, gamma=0.99):
        algo.policy.eval()
        obs = eval_env.reset()
        Q_pi = 0
        #init_obs = obs.reshape(1,-1)
        #act = algo.policy.sample_action(init_obs, deterministic=True)
        #Q_pi = algo.policy.Q_np(init_obs, act)
        total_reward = 0
        rs = []
        V_pi = 0
        done, i = False, 0
        while not done:
            # obs = (obs - mean)/std
            action = algo.policy.sample_action(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            rs.append(reward)
            i += 1
        if False:
            while i > 0:
                V_pi = V_pi * gamma + rs[i-1]
                i -= 1
        return total_reward, (Q_pi - V_pi)**2, Q_pi, V_pi

def sample_one_step(obs, env, algo, episode_steps, max_episode_steps):
    # TODO Update mean and std
    with torch.no_grad():
        algo.policy.eval()
        #obs = (obs - mean)/std
        action = algo.policy.sample_action(obs)
        next_obs, reward, done, _ = env.step(action)
        #reward = reward * Nr
        #algo.r_max = max(algo.r_max, reward)
        #algo.r_min = min(algo.r_min, reward)
        terminal = 0 if episode_steps == max_episode_steps else done
        algo.online_buffer.add(obs, next_obs, action, reward, terminal)
        algo.real_priority_buffer.add(obs, next_obs, action, reward, terminal)
        return next_obs, done

def sample_one_step_random(obs, env, algo, episode_steps, max_episode_steps):
    with torch.no_grad():
        algo.policy.eval()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        terminal = 0 if episode_steps == max_episode_steps else done
        algo.online_buffer.add(obs, next_obs, action, reward, terminal)
        return next_obs, done

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed=seed+42)

def save(dir ,filename, env_name, network_model):
    if not os.path.exists(dir):
        os.mkdir(dir)
    file = dir + env_name + "-" + filename 
    torch.save(network_model.state_dict(), file)
    print(f"***save the {network_model} model to {file}***")
    

def load(dir, filename, env_name, network_model):
    file = dir + env_name + "-" + filename
    if not os.path.exists(file):
        raise FileExistsError("Doesn't exist the model")
    network_model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    print(f"***load the model from {file}***")


def _gen_dir_name(taskname):
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{taskname}_{rand_str}'


class Log:
    def __init__(self, root_log_dir, cfg_dict='',
                 taskname = '',
                 txt_filename='config.txt',
                 csv_filename='progress.csv',
                 log_filename='LOGS.txt',
                 #cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name(taskname)
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        self.log_file = open(self.dir/log_filename, 'w')
        #(self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        #self.cfg_filename = cfg_filename
        self.flush = flush

    def write_config(self, args, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        configs = vars(args)
        for f in [sys.stdout, self.txt_file]:
            print(f'[{now_str}] '+'Running Configs = {', file=f, flush=self.flush)
            for k in configs.keys():
                print(f'\'{k}\': {configs[k]},', file=f, flush=self.flush)
            print("}", file=f, flush=self.flush)

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%m/%d-%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.log_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        self.log_file.close()
        if self.csv_file is not None:
            self.csv_file.close()