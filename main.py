"""
Implementation of SDV and SDV-FT

This implementation builds upon 
1) a PyTorch reproduction of the implementation of MOPO:
   https://github.com/junming-yang/mopo.
2) Author's implementation of POR in "A Policy-Guided Imitation
   Approach for Offline Reinforcement Learning":
   https://github.com/ryanxhr/POR.
3) Implementation of SDV:
   https://github.com/Misakau/SDV.
"""
import random

from matplotlib import pyplot as plt

import gym
import os
import d4rl
import numpy as np
import torch
from tqdm import trange, tqdm
from algo.PriorityBuffer import PriorityReplayBuffer

from sdv import SDV
from policy import GaussianPolicy
from value_functions import WeightNet, TwinQ, TwinV
from util import Normalizer, return_range, sample_one_step, set_seed, Log, sample_batch, torchify, evaluate_sdv

import time

from util import DEFAULT_DEVICE
import importlib
from algo.buffer import ReplayBuffer, myPriorityReplayBuffer
from algo.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mbpo import MBPO

def get_env_and_dataset(env_name, max_episode_steps, normalize, online=False):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

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

def main(args, logger):

    #torch.set_num_threads(1)

    env, dataset, mean, std, max_ret, min_ret = get_env_and_dataset(args.env_name,
                                                    args.max_episode_steps,
                                                    args.normalize
                                                    #,args.online
                                                    )

    r_max, r_min = dataset['rewards'].cpu().numpy().max(), dataset['rewards'].cpu().numpy().min()
    print(f"Single step rewards have range[{dataset['rewards'].min()},{dataset['rewards'].max()}], avg: {dataset['rewards'].mean()}")

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    task = args.env_name.split('-')[0]
    import_path = f"static_fns.{task}"
    if not args.fake_env:
        static_fns = importlib.import_module(import_path).StaticFns
    else:
        static_fns = importlib.import_module("static_fns.halfcheetah").StaticFns
    
    if args.type == 'sdv_s':
        goal_policy = GaussianPolicy(obs_dim + act_dim, obs_dim + 1, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,static_fns=static_fns,act_fnc='swish')
    else:
        goal_policy = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=2)
    
    goal_model = GaussianPolicy(obs_dim + act_dim, obs_dim + 1, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,static_fns=static_fns,act_fnc='swish')

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=dataset["observations"].shape[0],
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=10**6,
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )

    if args.online:
        online_buffer = ReplayBuffer(
            buffer_size=250*10**3,
            obs_dim=obs_dim,
            obs_dtype=np.float32,
            act_dim=act_dim,
            act_dtype=np.float32
        )
    else:
        online_buffer = None
    
    if args.online:
        fake_priority_buffer = PriorityReplayBuffer(
            buffer_size=int(2 ** np.ceil(np.log2(10**6))),
            obs_dim=obs_dim,
            act_dim=act_dim,
            p_max=1000
        )
    else:
        fake_priority_buffer = None

    if args.online:
        real_priority_buffer = myPriorityReplayBuffer(
            buffer_size=offline_buffer.max_size + 250000,
            obs_dim=obs_dim,
            obs_dtype=np.float32,
            act_dim=act_dim,
            act_dtype=np.float32
        )

        
    else:
        real_priority_buffer = None
    
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
    

    sdv = SDV(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=TwinV(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        goal_policy=goal_policy,
        goal_model=goal_model,
        max_model_steps=args.model_train_steps,
        tau=args.tau,
        alpha=args.alpha,
        discount=args.discount,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
    )

    # train sdv
    if not args.pretrain:
        algo_name = f"{args.type}_alpha-{args.alpha}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        os.makedirs(f"{args.log_dir}/{args.env_name}/{algo_name}", exist_ok=True)
        eval_log = open(f"{args.log_dir}/{args.env_name}/{algo_name}/seed-{args.seed}.txt", 'w')
        if not args.skip_model_train:
            # train guidance
            os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
            print(f"the models will be saved at {args.model_dir}/{args.env_name}/seed_{args.seed}_{algo_name}")
            
            for step in trange(args.model_train_steps):
                if args.type == 'sdv_t':
                    losses = sdv.myiql_update(**sample_batch(dataset, args.batch_size))
                    if args.only_model:
                        if step % args.max_episode_steps == 0:
                            logger.row(losses)
                elif args.type == 'sdv_b':
                    losses = sdv.no_adv_update(**sample_batch(dataset, args.batch_size))
                    if args.only_model:
                        if step % args.max_episode_steps == 0:
                            logger.row(losses)
                elif args.type == 'sdv_s':
                    losses = sdv.double_T(**sample_batch(dataset,args.batch_size*2))
                    if args.only_model:
                        if step % args.max_episode_steps == 0:
                            logger.row(losses)
                else:
                    raise(NotImplementedError)
            sdv.save(f"{args.model_dir}/{args.env_name}/seed_{args.seed}_{algo_name}")
        else:
            sdv.load(f"{args.model_dir}/{args.env_name}/{algo_name}")

    elif not args.pure:
        algo_name = f"{args.type}_alpha-{args.alpha}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        sdv.load(f"{args.model_dir}/{args.env_name}/{algo_name}")    
    if args.fake_env:
        sdv.plot_env(dataset['observations'],dataset['actions'],dataset['next_observations'],dataset['rewards'])
        env.predict(sdv.goal_policy,sdv.goal_model,dataset['observations'],dataset['next_observations'])
    
    weight_net = WeightNet(state_dim=obs_dim, act_dim=act_dim).to(DEFAULT_DEVICE)#WeightNet(state_dim=obs_dim, act_dim=act_dim).to(DEFAULT_DEVICE)
    
    mb_algo = MBPO(
        sac_policy,
        sdv.goal_model,
        sdv.goal_policy,
        weight_net,
        obs_mean=mean,
        obs_std=std,
        r_max=r_max,
        r_min=r_min,
        max_e=args.max_episode_steps,
        max_ret=max_ret,
        min_ret=min_ret,
        beta = args.beta,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        online_buffer=online_buffer,
        priority_buffer=(fake_priority_buffer, real_priority_buffer),
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=256,
        real_ratio=args.real_ratio,
        max_steps=args.decay_steps,
        model_type = args.type,
        update_model_fn=sdv.update_model
    )
    def eval_sdv(step, penalty=None, model_loss={}):
        #env.set_fine_tune(False)
        eval_returns = []
        for _ in range(args.n_eval_episodes):
            ret, _, _, _ = evaluate_sdv(env, mb_algo, mean, std)
            eval_returns.append(ret)
        eval_returns = np.array(eval_returns)
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
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
    if not args.pretrain:
        # train policy
        num_timesteps = 0
        train_epochs = args.train_steps // args.max_episode_steps if not args.fake_env else 100
        step_per_epoch = args.max_episode_steps if not args.fake_env else 100
        for e in range(1, train_epochs + 1):
            if args.only_model:
                break
            #mb_algo.model_buffer.clear() 
            mb_algo.policy.train()
            with tqdm(total=step_per_epoch, desc=f"Epoch #{e}/{train_epochs}") as t:
                p = None
                while t.n < t.total:
                    if num_timesteps % args.rollout_freq == 0:
                        p = mb_algo.rollout_transitions(e)
                    # update policy by sac
                    loss = mb_algo.learn_policy()
                    t.set_postfix(**loss)
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            if not args.fake_env:
                model_loss = {}
                average_returns = eval_sdv(num_timesteps,p, model_loss)
                eval_log.write(f'{num_timesteps + 1}\t{average_returns}\n')
                eval_log.flush()
                # save policy
                if args.trace_policy and e % 100 == 0:
                    torch.save(mb_algo.policy.state_dict(), os.path.join(args.model_dir,args.env_name, f"seed_{args.seed}_policy_iter{e}_alpha-{args.alpha}_normalize-{args.normalize}_type-{args.type}.pth"))
                else:
                    torch.save(mb_algo.policy.state_dict(), os.path.join(args.model_dir,args.env_name, f"seed_{args.seed}_policy_alpha-{args.alpha}_normalize-{args.normalize}_type-{args.type}.pth"))
        if args.fake_env:
            mb_algo.plot_qs()    
        eval_log.close()
    else:
        if args.pure:
            # ONLY ONLINE SAMPLES
            online_env = Normalizer(gym.make(args.env_name))
            online_env.init(env.mean, env.std, env.Nr)
            online_env.seed(seed=args.seed)
            print(online_env.mean)
            print(online_env.std)
            num_timesteps = 0
            train_epochs = args.train_steps // args.max_episode_steps
            step_per_epoch = args.max_episode_steps
            eval_sdv(num_timesteps)
            for e in range(0, train_epochs + 1):
                mb_algo.policy.train()
                done = False
                obs = online_env.reset()
                ep_step = 1
                print(f"Epoch #{e}/{train_epochs}")
                with tqdm(total=step_per_epoch, desc=f"Updating") as t:
                    while t.n < t.total:
                        if done:
                            obs = online_env.reset()
                            ep_step = 1
                        # Interact with environment
                        obs, done = sample_one_step(obs, online_env, mb_algo, ep_step, args.max_episode_steps)
                        num_timesteps += 1
                        ep_step += 1
                        # SAC update
                        if num_timesteps > 1000:
                            loss = mb_algo.learn_pure_policy()
                            t.set_postfix(**loss)
                        t.update(1)
                # Evaluate current policy
                if e > 1:
                    eval_sdv(num_timesteps)
                # Save policy
                torch.save(mb_algo.policy.state_dict(), os.path.join(args.model_dir,args.env_name, f"seed_{args.seed}_normalize-{args.normalize}_SACPolicy.pth"))
        elif args.online:
            model_dict = torch.load(os.path.join(args.model_dir,args.env_name, f"policy_alpha-{args.alpha}_normalize-{args.normalize}_type-{args.type}.pth"),map_location=DEFAULT_DEVICE)
            mb_algo.policy.load_state_dict(model_dict)
            print(model_dict.keys())
            print("ONLINE TRAINNING!")
            online_env = Normalizer(gym.make(args.env_name))
            online_env.init(env.mean, env.std, env.Nr)
            online_env.seed(seed=args.seed)
            print(online_env.mean)
            print(online_env.std)
            print("Load offline data......")
            mb_algo.real_priority_buffer.load_offline(offline_buffer)
            num_timesteps = 0
            eval_sdv(num_timesteps)
            train_epochs = args.train_steps // args.max_episode_steps
            step_per_epoch = args.max_episode_steps
            model_step_per_epoch = 1000
            off_rate = -1
            real_ratio = args.real_ratio
            real_ratio_delta = (1 - args.real_ratio) / args.decay_steps
            priority = 1.0
            pweight = 1.0
            # mb_algo.model_buffer.clear()
            for e in range(1, train_epochs + 1):
                
                #mb_algo.model_buffer.clear() 
                mb_algo.policy.train()
                #env.set_fine_tune(True)
                done = False
                obs = online_env.reset()
                ep_step = 1
                # generate fake transitions
                if real_ratio < 1:
                    p = mb_algo.rollout_transitions(e)

                mb_algo.update_time_priority(priority)
                
                print(f"Epoch #{e}/{train_epochs}")
                # get online data 
                with tqdm(total=step_per_epoch, desc=f"Online Sampling") as t:
                    while t.n < t.total:
                        if done:
                            obs = online_env.reset() 
                            ep_step = 1
                        obs, done = sample_one_step(obs, online_env, mb_algo, ep_step, args.max_episode_steps)
                        num_timesteps += 1
                        ep_step += 1
                        t.update(1)
                
                if args.update_model:
                    # update model
                    with tqdm(total=model_step_per_epoch, desc=f"Model updates") as t:
                        sdv.reset_scheduler(model_step_per_epoch)
                        while t.n < t.total:
                            losses = mb_algo.update_model()
                            t.update(1)
                            t.set_postfix(**losses)
                
                # off-policy updates
                off_update = step_per_epoch if e > 1 else 5 * step_per_epoch
                with tqdm(total=off_update, desc=f"SAC updates") as t:
                    while t.n < t.total:
                        # update policy by SAC
                        if not args.online_sample:
                            if args.wobr:
                                loss = mb_algo.learn_nobr_policy(fake_ratio=1-real_ratio)
                            else:
                                loss = mb_algo.learn_online_policy(
                                                                use_priority=True,
                                                                fake_ratio=1-real_ratio,
                                                                priority=priority)
                                off_rate = loss['off_rate']
                        else:
                            loss = mb_algo.learn_pure_policy()
                        t.set_postfix(**loss)
                        t.update(1)
               
                real_ratio = min(1, real_ratio + real_ratio_delta)

                pweight = mb_algo.get_offline_priority()
                priority = min(1.0 / (1.5*e), pweight)
                # evaluate current policy
                if e >= 1: eval_sdv(num_timesteps, (priority,off_rate))
                # save policy
                pathname = os.path.join(args.model_dir,args.env_name, f"seed_{args.seed}_OnlinePolicy_alpha-{args.alpha}_normalize-{args.normalize}_type-{args.type}.pth")
                torch.save(mb_algo.policy.state_dict(), pathname)   

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="hopper-medium-replay-v2")
    parser.add_argument('--root_log_dir', type=str, default="./exp_log")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=1) #try 13
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--pretrain_steps', type=int, default=5*10**5)
    parser.add_argument('--model_train_steps', type=int, default=5*10**5)
    parser.add_argument('--train_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--eval_period', type=int, default=1000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--type", type=str, choices=['sdv_t','sdv_s', 'sdv_b'], default='sdv_t')
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument("--fake_env", action='store_true')
    parser.add_argument("--skip_model_train", action='store_true')
    parser.add_argument('--only_model', action='store_true')
    parser.add_argument('--reward_norm', action='store_true')
    parser.add_argument('--pure', action='store_true')
    parser.add_argument('--online_sample', action='store_true')
    parser.add_argument('--lamda_decay', action='store_true')
    parser.add_argument('--decay_steps', type=int, default=5)
    parser.add_argument('--trace_policy', action='store_true')
    parser.add_argument('--update_model', action='store_true')
    parser.add_argument('--wobr', action='store_true')
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
    logger = Log(root_log_dir=args.root_log_dir,taskname=args.env_name)
    logger.write_config(args)
    main(args, logger)
    logger.close()