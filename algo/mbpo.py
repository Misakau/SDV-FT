from matplotlib import pyplot as plt
import numpy as np
import torch
from util import torchify

class MBPO:
    def __init__(
            self,
            policy,
            dynamics_model,
            goal_policy,
            weight_net,
            obs_mean,
            obs_std,
            r_max,
            r_min,
            max_e,
            max_ret,
            min_ret,
            beta,
            offline_buffer,
            model_buffer,
            online_buffer,
            priority_buffer,
            reward_penalty_coef,
            rollout_length,
            batch_size,
            real_ratio,
            max_steps=10,
            rollout_batch_size=50000,
            model_type = 'sdv_t',
            update_model_fn=None
    ):
        self.policy = policy
        self.goal_policy = goal_policy
        self.dynamics_model = dynamics_model
        self.weight_net = weight_net
        self.model_type = model_type
        if self.weight_net is not None:
            self.weight_net_optimizer = torch.optim.Adam(self.weight_net.parameters(), lr=3e-4)
        else:
            self.weight_net_optimizer = None
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self.online_buffer = online_buffer
        self.fake_priority_buffer = priority_buffer[0]
        self.real_priority_buffer = priority_buffer[1]
        self._reward_penalty_coef_start = reward_penalty_coef
        self._reward_penalty_coef_delta = reward_penalty_coef / max_steps
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._decay_rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.r_max = r_max
        self.r_min = r_min
        self.max_e=max_e
        self.max_ret=max_ret
        self.min_ret=min_ret
        self.beta = beta
        self.update_model_fn = update_model_fn
        self.offline_prioriy = 1.0

    def _sample_initial_transitions(self, timestep, online=False):
        if online:
            data = self.real_priority_buffer.sample(self._decay_rollout_batch_size)
            return data
        else:
            return self.offline_buffer.sample(self._decay_rollout_batch_size)
    
    def get_penalty(self, observations, next_observations, actions, online=False):
        dists, mystd = 1, 1
        with torch.no_grad():
            if online:
                _, obs_vars, _, rew_var = self.dynamics_model.get_mean_var(observations, actions)
                mystd = np.sqrt(np.sum(np.concatenate([obs_vars, rew_var],axis=1),axis=1,keepdims=True))
            else:
                if self.model_type == 'sdv_t':
                    goal_obs_t = self.goal_policy.act(torchify(observations), deterministic=True)
                else:
                    goal_obs_t = self.goal_policy.act(torch.cat([torchify(observations), torchify(actions)],dim=1), deterministic=True)[:,:-1]
                goal_obs = goal_obs_t.cpu().numpy()
                dists = (goal_obs - next_observations)*(goal_obs - next_observations)

                dists = np.sqrt(np.sum(dists,axis=1,keepdims=True))
            return dists * mystd
        
    def rollout_transitions(self, timestep, decay=False, online=False):
        self._decay_rollout_batch_size = self._rollout_batch_size
        init_transitions = self._sample_initial_transitions(timestep,online)
        # rollout
        observations = init_transitions["observations"]
        p = 0
        with torch.no_grad():
            for _ in range(self._rollout_length):
                actions = self.policy.sample_action(observations)
                r_max = self.r_max
                r_min = self.r_min
                next_observations, rewards, terminals = self.dynamics_model.predict(observations, actions, self.obs_mean, self.obs_std, r_max, r_min)
                
                #rewards reshape
                penalty = self.get_penalty(observations, next_observations, actions)
                rewards -= self._reward_penalty_coef * penalty
                
                self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
                if self.fake_priority_buffer is not None:
                    for i in range(observations.shape[0]):
                        self.fake_priority_buffer.add_sample(
                            observations[i],
                            actions[i],
                            rewards[i],
                            terminals[i],
                            next_observations[i],
                            'fake'
                        )
                nonterm_mask = (~terminals).flatten()
                if nonterm_mask.sum() == 0:
                    break
                observations = next_observations[nonterm_mask]
                p += penalty.mean()
        
        if decay: 
            self._reward_penalty_coef = max(0, self._reward_penalty_coef - self._reward_penalty_coef_delta)
            #self._reward_penalty_coef = self._reward_penalty_coef_start / timestep   
        return p / self._rollout_length

    def sample_data_for_finetune(self, size, offline_rate):
        offline_size = int(size * offline_rate)
        online_size = size - offline_size
        offline_batch = self.offline_buffer.sample(batch_size=offline_size)
        online_batch = self.online_buffer.sample(batch_size=online_size)
        data = {
            "observations": np.concatenate([offline_batch["observations"], online_batch["observations"]], axis=0),
            "actions": np.concatenate([offline_batch["actions"], online_batch["actions"]], axis=0),
            "next_observations": np.concatenate([offline_batch["next_observations"], online_batch["next_observations"]],
                                                axis=0),
            "terminals": np.concatenate([offline_batch["terminals"], online_batch["terminals"]], axis=0),
            "rewards": np.concatenate([offline_batch["rewards"], online_batch["rewards"]], axis=0)
        }
        return data

    def sample_data_for_train(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]],
                                                axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        return data
    
    def learn_policy(self):

        data = self.sample_data_for_train()

        loss = self.policy.learn(data)

        return loss

    def learn_br_policy(self):
        data = self.real_priority_buffer.random_batch(self._batch_size)
        self.policy.train()
        loss = self.policy.learn(data)
        priority_loss = self.learn_priority()
        self.update_priority(data, self.real_priority_buffer)
        return loss

    def learn_pure_policy(self):

        data = self.online_buffer.sample(self._batch_size)

        loss = self.policy.learn(data)

        return loss

    def update_model(self):
        if self.update_model_fn is None:
            raise NotImplementedError()
        #data = self.sample_uniform_real_data(self._batch_size)
        data = self.real_priority_buffer.sample(self._batch_size)
        model_loss = self.update_model_fn(
            data['observations'],
            data['actions'],
            data['next_observations'],
            data['rewards'],
            data['terminals']
        )
        #self.update_time_priority(data, priority)
        return model_loss
    
    def update_time_priority(self, priority):
        self.real_priority_buffer.update_offline_priority(priority)
    
    def update_all_priority(self, off_p, on_p):
        self.real_priority_buffer.update_all_priority(off_p, on_p)

    def update_real_priority(self, data, offline_priority):
        #self.sample_data_for_train() #
        offline_batch = self.offline_buffer.sample(self._batch_size)
        offline_obs = offline_batch["observations"]
        offline_actions = offline_batch["actions"]
        
        self.weight_net.eval()
        with torch.no_grad():
            offline_weight = self.weight_net(
                torchify(offline_obs), torchify(offline_actions)
            )
            weight = self.weight_net(
                torchify(data['observations']), torchify(data['actions'])
            )
            normalized_weight = weight ** (1/5.0) / ((offline_weight ** (1/5.0)).mean() + 1e-10)

        new_priority = normalized_weight.clamp(0.001, 1000).detach().cpu().numpy()

        self.real_priority_buffer.update_priority(
            data["idx"].squeeze(),
            new_priority.squeeze(),
        )

    def update_priority(self, data, priority_buffer):
        #self.sample_data_for_train() #
        offline_batch = self.offline_buffer.sample(self._batch_size)
        offline_obs = offline_batch["observations"]
        offline_actions = offline_batch["actions"]
        
        self.weight_net.eval()
        with torch.no_grad():
            offline_weight = self.weight_net(
                torchify(offline_obs), torchify(offline_actions)
            )
            weight = self.weight_net(
                torchify(data['observations']), torchify(data['actions'])
            )
            normalized_weight = weight ** (1/5.0) / ((offline_weight ** (1/5.0)).mean() + 1e-10)

        new_priority = normalized_weight.clamp(0.001, 1000).detach().cpu().numpy()

        priority_buffer.update_priorities(
            data["tree_idxs"].squeeze().astype(int),
            new_priority.squeeze(),
        )

    def get_offline_priority(self):
        #self.sample_data_for_train() #
        offline_batch = self.offline_buffer.sample(1024)
        offline_obs = offline_batch["observations"]
        offline_actions = offline_batch["actions"]
        
        self.weight_net.eval()
        with torch.no_grad():
            offline_weight = self.weight_net(
                torchify(offline_obs), torchify(offline_actions)
            )
            offline_weight = offline_weight ** (1/5.0)
        new_priority = (offline_weight.clamp(0.001, 1)).detach().cpu().numpy()
        return new_priority.mean()
    
    def get_online_priority(self, offline_p):

        online_batch = self.online_buffer.sample(1024)
        online_obs = online_batch["observations"]
        online_actions = online_batch["actions"]
        
        self.weight_net.eval()
        with torch.no_grad():
            online_weight = self.weight_net(
                torchify(online_obs), torchify(online_actions)
            )
            online_weight = online_weight ** (1/5.0)

        online_p = (online_weight.clamp(1, 1000)).detach().mean().cpu().numpy()

        offline_length = offline_p * self.offline_buffer.size
        online_length = online_p * self.online_buffer.size
        new_priority = offline_length / online_length

        return new_priority.mean()
    
    def learn_priority(self):
        self.weight_net.train()
        #self.sample_data_for_train()#
        offline_batch = self.offline_buffer.sample(self._batch_size)
        online_batch = self.online_buffer.sample(self._batch_size)
        offline_obs = offline_batch['observations']
        offline_actions = offline_batch['actions']
        online_obs = online_batch['observations']
        online_actions = online_batch['actions']
        # weight network loss calculation!
        offline_weight = self.weight_net(
            torchify(offline_obs), torchify(offline_actions)
        )

        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)

        online_weight = self.weight_net(
            torchify(online_obs), torchify(online_actions)
        )
        online_f_prime = torch.log(2 * online_weight / (online_weight + 1) + 1e-10)

        weight_loss = (offline_f_star - online_f_prime).mean()

        self.weight_net_optimizer.zero_grad()

        weight_loss.backward()

        self.weight_net_optimizer.step()
        
        return weight_loss.item()
    
    def sample_uniform_real_data(self, size):
        idx = np.random.randint(low=0,
                                high=self.offline_buffer.size + self.online_buffer.size,
                                size=size
                                )
        offline_batch = self.offline_buffer.get(idx[idx<self.offline_buffer.size])
        online_batch = self.online_buffer.get(idx[idx>=self.offline_buffer.size] - self.offline_buffer.size)
        
        offline_obs = offline_batch["observations"]
        offline_actions = offline_batch["actions"]
        offline_rewards = offline_batch["rewards"]
        offline_next_obs = offline_batch["next_observations"]
        offline_terminals = offline_batch["terminals"]

        online_obs = online_batch["observations"]
        online_actions = online_batch["actions"]
        online_rewards = online_batch["rewards"]
        online_next_obs = online_batch["next_observations"]
        online_terminals = online_batch["terminals"]
        
        data = {
            "observations": np.concatenate((offline_obs, online_obs), axis=0),
            "actions": np.concatenate((offline_actions, online_actions), axis=0),
            "next_observations": np.concatenate((offline_next_obs, online_next_obs), axis=0),
            "rewards": np.concatenate((offline_rewards, online_rewards), axis=0),
            "terminals": np.concatenate((offline_terminals, online_terminals), axis=0)
        }
        return data
    
    def learn_online_policy(self, use_priority=False, fake_ratio=0.5, priority=1.0):
        data = {}
        
        fake_size = int(self._batch_size*fake_ratio)
        real_size = self._batch_size - fake_size
        real_batch = self.real_priority_buffer.sample(real_size)
        if fake_size > 0:
            
            fake_batch = self.fake_priority_buffer.random_batch(fake_size) if use_priority else self.model_buffer.sample(fake_size)
            data = {
                'observations': np.concatenate((real_batch['observations'], fake_batch['observations']), axis=0),
                'rewards': np.concatenate((real_batch['rewards'], fake_batch['rewards']), axis=0),
                'actions': np.concatenate((real_batch['actions'], fake_batch['actions']), axis=0),
                'terminals': np.concatenate((real_batch['terminals'], fake_batch['terminals']), axis=0),
                'next_observations': np.concatenate((real_batch['next_observations'], fake_batch['next_observations']), axis=0),
            }
        else:
            data = real_batch

        self.policy.train()
        loss = self.policy.learn(data)
        priority_loss = self.learn_priority()
        
        if fake_size > 0 and use_priority:
            self.update_priority(fake_batch, self.fake_priority_buffer)
        
        self.update_real_priority(real_batch, priority)
        loss['off_rate'] = real_batch['types'].sum() / real_size
        loss['ploss'] = priority_loss
        return loss
    
    def learn_nobr_policy(self, fake_ratio=0.5):
        data = {}
        fake_size = int(self._batch_size*fake_ratio)
        real_size = self._batch_size - fake_size
        real_batch = self.real_priority_buffer.sample(real_size)

        if fake_size > 0:
            fake_batch = self.fake_priority_buffer.random_batch(fake_size)
            data = {
                'observations': np.concatenate((real_batch['observations'], fake_batch['observations']), axis=0),
                'rewards': np.concatenate((real_batch['rewards'], fake_batch['rewards']), axis=0),
                'actions': np.concatenate((real_batch['actions'], fake_batch['actions']), axis=0),
                'terminals': np.concatenate((real_batch['terminals'], fake_batch['terminals']), axis=0),
                'next_observations': np.concatenate((real_batch['next_observations'], fake_batch['next_observations']), axis=0),
            }
        else:
            data = real_batch

        self.policy.train()
        loss = self.policy.learn(data)
        priority_loss = self.learn_priority()
        
        if fake_size > 0:
            self.update_priority(fake_batch, self.fake_priority_buffer)
        loss['f'] = 1 - fake_ratio
        loss['ploss'] = priority_loss
        return loss
    
    def clone_policy(self):
        real_batch = self.offline_buffer.sample(batch_size=self._batch_size)
        data = {
            "observations": real_batch["observations"],
            "actions": real_batch["actions"],
            "next_observations": real_batch["next_observations"],
            "terminals": real_batch["terminals"],
            "rewards": real_batch["rewards"],
        }
        loss = self.policy.behavior_clone(data)
        return loss
    
    def plot_qs(self):
        real_batch = self.offline_buffer.sample(batch_size=self._batch_size)
        act = np.arange(-1,1,0.01).reshape(-1,1)
        obs = real_batch["observations"][:act.shape[0]]
        qout = self.policy.Q_np(obs,act)
        plt.figure()
        plt.ylabel('Q_sac(s,a)')
        plt.xlabel('a')
        maxindx = qout.argmax()
        plt.plot(act[:,0],qout[:,0],c='r')
        plt.plot([act[maxindx],act[maxindx]],[-4,2],c='black')
        plt.plot()
        plt.savefig("./fake_env_qsac_por.png")
        plt.close()
