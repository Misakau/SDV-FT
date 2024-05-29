"""
SAC
"""
import gym
import os
import d4rl
import numpy as np
import torch
from tqdm import tqdm
from util import sample_one_step, sample_one_step_random, set_seed, Log, evaluate_sdv

import time

from util import DEFAULT_DEVICE
from algo.buffer import ReplayBuffer
from algo.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mbpo import MBPO

def get_env_and_dataset(env_name):
    env = gym.make(env_name)
    return env

def main(args, logger):

    #torch.set_num_threads(1)

    env = get_env_and_dataset(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    set_seed(args.seed, env=env)

    # create buffer
    offline_buffer = None

    online_buffer = ReplayBuffer(
        buffer_size=250*10**3,
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )

    model_buffer = None
    fake_priority_buffer = None
    
    real_priority_buffer = ReplayBuffer(
        buffer_size=250*10**3,
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )
    
    # create SAC policy model
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=act_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, DEFAULT_DEVICE)
    critic1 = Critic(critic1_backbone, DEFAULT_DEVICE)
    critic2 = Critic(critic2_backbone, DEFAULT_DEVICE)
    
    # auto-alpha
    sac_target_entropy = -act_dim
    sac_log_alpha = torch.zeros(1, requires_grad=True, device=DEFAULT_DEVICE)
    sac_alpha_optim = torch.optim.Adam([sac_log_alpha], lr=args.sac_alpha_lr)

    # create SAC policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_lr=args.sac_actor_lr,
        critic_lr=args.sac_critic_lr,
        action_space=env.action_space,
        dist=dist,
        tau=args.sac_tau,
        gamma=args.sac_gamma,
        alpha=(sac_target_entropy, sac_log_alpha, sac_alpha_optim),
        device=DEFAULT_DEVICE
    )

    mb_algo = MBPO(
        sac_policy,
        None,
        None,
        None,
        obs_mean=0,
        obs_std=0,
        r_max=0,
        r_min=0,
        max_e=args.max_episode_steps,
        max_ret=0,
        min_ret=0,
        beta = args.beta,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        online_buffer=online_buffer,
        priority_buffer=(fake_priority_buffer, real_priority_buffer),
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=256,
        real_ratio=args.real_ratio,
        update_model_fn=None
    )

    def eval_sdv(step, penalty=None, model_loss={}):
        eval_returns = []
        for _ in range(args.n_eval_episodes):
            ret, _, _, _ = evaluate_sdv(env, mb_algo)
            eval_returns.append(ret)
        eval_returns = np.array(eval_returns)
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        #print(f"return mean: {eval_returns.mean()},normalized return mean: {normalized_returns.mean()},Q error: {eval_errors.mean()}")
        msg = {
            'step': step,
            'return mean': eval_returns.mean(),
            'normalized return mean': normalized_returns.mean(),
            'dist/pweight': penalty
        }
        for k, v in model_loss.items():
            msg[k] = v
        logger.row(msg)
        return normalized_returns.mean()
    
    # SAC
    print("ONLINE TRAINNING!")
    online_env = gym.make(args.env_name)
    online_env.seed(seed=args.seed)
    num_timesteps = 0
    eval_sdv(num_timesteps)
    train_epochs = args.train_steps // args.max_episode_steps
    step_per_epoch = args.max_episode_steps
    for e in range(1, train_epochs + 1):
        mb_algo.policy.train()
        done = False
        obs = online_env.reset()
        print(f"Epoch #{e}/{train_epochs}")
        ep_step = 1
        # get online data 
        with tqdm(total=step_per_epoch, desc=f"Online Sampling") as t:
            while t.n < t.total:
                if done: 
                    obs = online_env.reset()
                    ep_step = 1
                # Interact with environment
                if num_timesteps > 1000:
                    obs, done = sample_one_step(obs, online_env, mb_algo, ep_step, args.max_episode_steps)
                else:
                    obs, done = sample_one_step_random(obs, online_env, mb_algo, ep_step, args.max_episode_steps)
                num_timesteps += 1
                ep_step += 1
                # SAC update
                if num_timesteps > 1000:
                    for _ in range(args.utd):
                        loss = mb_algo.learn_pure_policy()
                        t.set_postfix(**loss)
                t.update(1)
                # Evaluate current policy
            eval_sdv(num_timesteps)
        # Save policy
        torch.save(mb_algo.policy.state_dict(), os.path.join(args.model_dir,args.env_name, f"seed_{args.seed}_normalize-{args.normalize}_SACPolicy.pth"))  

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="hopper-medium-replay-v2")
    parser.add_argument('--log_dir', type=str, default="./fixpuresaclog")
    parser.add_argument('--model_dir', type=str, default="./othermodels/")
    parser.add_argument('--seed', type=int, default=1) #try 13
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--train_steps', type=int, default=250000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--utd', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--mle_test', action='store_true')
    parser.add_argument('--pure', action='store_true')
    # SAC
    parser.add_argument("--sac_actor_lr", type=float, default=3e-4)
    parser.add_argument("--sac_critic_lr", type=float, default=3e-4)
    parser.add_argument("--sac_gamma", type=float, default=0.99)
    parser.add_argument("--sac_tau", type=float, default=0.005)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)

    # MBPO
    parser.add_argument("--rollout_freq", type=int, default=1000)
    parser.add_argument("--rollout_length", type=int, default=5)
    parser.add_argument("--real_ratio", type=float, default=0.5)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--reward_penalty_coef", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0)
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()
    logger = Log(root_log_dir=args.log_dir,taskname='SAC'+args.env_name)
    logger.write_config(args)
    main(args, logger)
    logger.close()