"""
SAC
"""
import os
import numpy as np
import torch
from mymaze import myEnv
from util import set_seed, Log
import time
from util import DEFAULT_DEVICE
from algo.buffer import ReplayBuffer
from algo.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy

def get_env_and_dataset():
    env = myEnv(render=False)
    return env
    
def main(args, logger):

    eval_env = get_env_and_dataset()
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    print(obs_dim)
    print(act_dim)

    set_seed(args.seed)

    online_buffer = ReplayBuffer(
        buffer_size=50000,
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
        action_space=eval_env.action_space,
        dist=dist,
        tau=args.sac_tau,
        gamma=args.sac_gamma,
        alpha=(sac_target_entropy, sac_log_alpha, sac_alpha_optim),
        device=DEFAULT_DEVICE
    )

    def eval_policy(t):
        eval_returns = []
        for i in range(5):
            sac_policy.eval()
            obs = eval_env.reset()
            total_reward = 0
            done= False
            while not done:
                action = sac_policy.sample_action(obs)
                obs, reward, done = eval_env.step(action)
                total_reward += reward
            eval_returns.append(total_reward)
        eval_returns = np.array(eval_returns).mean()
        print(f'Eval Return at {t}: {eval_returns}')
        return eval_returns
    
    # SAC
    print("SIMPLE ENV TRAINNING!")
    online_env = myEnv()
    num_timesteps = 0
    step_per_epoch = 1000000
    done = False
    obs = online_env.reset()
    for e in range(1, step_per_epoch + 1):
        # get online data 
        if done: obs = online_env.reset()
        # Interact with environment    
        with torch.no_grad():
            sac_policy.eval()
            action = sac_policy.sample_action(obs)
            next_obs, reward, terminal = online_env.step(action)
            #print(f"agent at {obs}, action {action}, reward {reward}, terminal {terminal}")
            online_buffer.add(obs, next_obs, action, reward, terminal)
            obs, done = next_obs, terminal
            
        # SAC update
        sac_policy.train()
        data = online_buffer.sample(args.batch_size)
        sac_policy.learn(data)

        # Evaluate current policy
        num_timesteps += 1
        if num_timesteps % 5000 == 0:
            eval_policy(num_timesteps)
            # Save policy
            torch.save(sac_policy.state_dict(), os.path.join("./myenv", f"Policy.pth"))  

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Common
    parser.add_argument('--env_name', type=str, default="myenv")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=1) #try 13
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)

    # SAC
    parser.add_argument("--sac_actor_lr", type=float, default=3e-4)
    parser.add_argument("--sac_critic_lr", type=float, default=3e-4)
    parser.add_argument("--sac_gamma", type=float, default=0.99)
    parser.add_argument("--sac_tau", type=float, default=0.005)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)

    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()
    logger = Log(root_log_dir='./motilog',taskname='moti'+args.env_name)
    #logger.write_config(args)
    main(args, logger)
    logger.close()