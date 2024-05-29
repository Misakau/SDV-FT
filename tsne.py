import random

from matplotlib import pyplot as plt

import gym
import d4rl
import numpy as np
import torch
from policy import GaussianPolicy
from util import Normalizer, return_range, set_seed, torchify
from tsne_torch import TorchTSNE as TSNE
import time

import seaborn as sns
from util import DEFAULT_DEVICE

def get_env_and_dataset(env_name, max_episode_steps, normalize, online=False):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    Nr = 1
    if online:
        obs_n = dataset['observations'].shape[0]
        idxs = random.sample(range(0,obs_n),min(obs_n,5*10**4))
        for k in dataset.keys():
            dataset[k] = dataset[k][idxs]
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        print(f'max score: {d4rl.get_normalized_score(args.env_name, max_ret) * 100.0}')
        if args.reward_norm:
            Nr = max_episode_steps / (max_ret - min_ret)
            dataset['rewards'] /= (max_ret - min_ret)
            dataset['rewards'] *= max_episode_steps
    else:
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        print(f'max score: {d4rl.get_normalized_score(args.env_name, max_ret) * 100.0}')

    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:
        mean = dataset['observations'].mean(0)
        std = dataset['observations'].std(0) + 1e-3
        dataset['observations'] = (dataset['observations'] - mean)/std
        dataset['next_observations'] = (dataset['next_observations'] - mean)/std
        env = Normalizer(env)
        env.init(mean, std)

        if any(s in env_name for s in ('fake_env')):
            plt.figure()
            plt.ylabel('T(s\'|s,a)')
            plt.xlabel('a')
            plt.scatter(dataset['actions'][:,0],dataset['observations'][:,0],s=0.01)
            plt.scatter(dataset['actions'][:,0],dataset['rewards'],s=0.01)
            plt.savefig("./normalized_fake_env.png")
            plt.close()
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)
        env = Normalizer(env)
        env.init(mean, std)
    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, mean, std, max_ret, min_ret

def main(args):

    #torch.set_num_threads(1)

    env, dataset, _, _, _, _ = get_env_and_dataset(args.env_name,
                                                    args.max_episode_steps,
                                                    args.normalize
                                                    )
    print(f"Single step rewards have range[{dataset['rewards'].min()},{dataset['rewards'].max()}], avg: {dataset['rewards'].mean()}")
    
    N = dataset['observations'].shape[0]
    S = 5000
    print(f'Sample {S} from {N}')
    
    obs_dim = dataset['observations'].shape[1]
    set_seed(args.seed, env=env)

    goal_model = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=2).to(DEFAULT_DEVICE)

    model_name = f"{args.model_dir}/{args.env_name}/{args.type}_goal_network"
    goal_model.load_state_dict(torch.load(model_name, map_location=DEFAULT_DEVICE))

    index = torch.LongTensor(random.sample(range(N), S)).to(DEFAULT_DEVICE)
    rewards = 0.5*(dataset['rewards'] - dataset['rewards'].min()) / (dataset['rewards'].max() - dataset['rewards'].min())
    rewards_real =  torch.index_select(rewards,0,index)
    observations_real = torch.index_select(dataset['observations'],0,index)
    next_observations_real = torch.index_select(dataset['next_observations'],0,index)
    #print(observations_real)
    next_observations_fake = goal_model.act(observations_real,True)
    inputs = torch.concat([next_observations_real,next_observations_fake])
    outputs = TSNE(initial_dims=obs_dim,verbose=True).fit_transform(inputs)
    X_real, X_model, real_r = outputs[:S], outputs[S+1:], rewards_real.cpu().numpy()

    plt.figure()
    plt.title(args.env_name)
    plt.xlabel('x')
    plt.ylabel('y')
    ax = sns.kdeplot(x=X_model[:,0], y=X_model[:,1], fill=True, cmap='Spectral_r', cbar=True, cbar_kws={'label':'$g_\omega(s\',s)$'})
    plt.scatter(X_real[:,0],X_real[:,1],s=3,color='m',alpha=real_r,label='Real observation')
    ax.legend()
    plt.legend()
    plt.savefig(f"{args.env_name}_{args.type}_tsne.pdf")
    plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="hopper-medium-v2")
    parser.add_argument('--root_log_dir', type=str, default="./exp_log")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=1) #try 13
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--type", type=str, choices=['sdv_t','sdv_b'], default='sdv_t')
    parser.add_argument('--reward_norm', action='store_true')

    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()
    for e in ['hopper-random-v2','hopper-medium-v2','hopper-expert-v2']:
        args.env_name=e
        main(args)
        torch.cuda.empty_cache()