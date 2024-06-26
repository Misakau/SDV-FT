import numpy as np
import torch
from util import DEFAULT_DEVICE

class ReplayBuffer:
    """
    the type of datas in buffer is np.ndarry
    """
    def __init__(
            self,
            buffer_size,
            obs_dim,
            obs_dtype,
            act_dim,
            act_dtype,
    ):
        self.max_size = buffer_size
        self.obs_dim = obs_dim
        self.obs_dtype = obs_dtype
        self.act_dim = act_dim
        self.act_dtype = act_dtype

        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((self.max_size, self.obs_dim), dtype=obs_dtype)
        self.next_observations = np.zeros((self.max_size, self.obs_dim), dtype=obs_dtype)
        self.actions = np.zeros((self.max_size, self.act_dim), dtype=act_dtype)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self.max_size, 1), dtype=np.float32)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, terminal):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset):
        observations = dataset["observations"].detach().cpu().numpy().astype(self.obs_dtype)
        next_observations = dataset["next_observations"].detach().cpu().numpy().astype(self.obs_dtype)
        actions = dataset["actions"].detach().cpu().numpy().astype(self.act_dtype)
        rewards = dataset["rewards"].detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        terminals = dataset["terminals"].detach().cpu().numpy().astype(np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self.ptr = len(observations) % self.max_size
        self.size = len(observations)

    def add_batch(self, obs, next_obs, actions, rewards, terminals):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = np.array(obs[:first_add_size]).copy()
            self.next_observations[begin:end] = np.array(next_obs[:first_add_size]).copy()
            self.actions[begin:end] = np.array(actions[:first_add_size]).copy()
            self.rewards[begin:end] = np.array(rewards[:first_add_size]).copy()
            self.terminals[begin:end] = np.array(terminals[:first_add_size]).copy()

            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = np.array(obs[first_add_size:]).copy()
            self.next_observations[begin:end] = np.array(next_obs[first_add_size:]).copy()
            self.actions[begin:end] = np.array(actions[first_add_size:]).copy()
            self.rewards[begin:end] = np.array(rewards[first_add_size:]).copy()
            self.terminals[begin:end] = np.array(terminals[first_add_size:]).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = np.array(obs).copy()
            self.next_observations[begin:end] = np.array(next_obs).copy()
            self.actions[begin:end] = np.array(actions).copy()
            self.rewards[begin:end] = np.array(rewards).copy()
            self.terminals[begin:end] = np.array(terminals).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def get(self, batch_indices):
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy()
        }

    def sample(self, batch_size):
        batch_indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy()
        }

    def sample_all(self):
        return {
            "observations": self.observations[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "next_observations": self.next_observations[:self.size].copy(),
            "terminals": self.terminals[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy()
        }

    @property
    def get_size(self):
        return self.size

class myPriorityReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_dim, obs_dtype, act_dim, act_dtype):
        super().__init__(buffer_size, obs_dim, obs_dtype, act_dim, act_dtype)
        self.prioritys = np.zeros((self.max_size, 1), dtype=np.float32)
        self.types = np.zeros((self.max_size, 1), dtype=np.int8)
        self.timestamps = np.zeros((self.max_size, 1), dtype=np.float32)
        self.max_p = 1.0
    
    def load_offline(self, offline_buffer):
        offline_data = offline_buffer.sample_all()
        offline_size = offline_buffer.size
        types = np.ones((offline_size,1),dtype=np.int8)
        timestamps = np.ones((offline_size, 1), dtype=np.float32)
        self.add_batch(
            offline_data['observations'],
            offline_data['next_observations'],
            offline_data['actions'],
            offline_data['rewards'],
            offline_data['terminals'],
            types,
            timestamps)
        self.max_p = offline_size / 1000

    def add(self, obs, next_obs, action, reward, terminal, type=0, priority=1.0):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()
        self.types[self.ptr] = type
        self.timestamps[self.ptr] = 1
        self.prioritys[self.ptr] = self.max_p

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, next_obs, actions, rewards, terminals, types, timestamps):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = np.array(obs[:first_add_size]).copy()
            self.next_observations[begin:end] = np.array(next_obs[:first_add_size]).copy()
            self.actions[begin:end] = np.array(actions[:first_add_size]).copy()
            self.rewards[begin:end] = np.array(rewards[:first_add_size]).copy()
            self.terminals[begin:end] = np.array(terminals[:first_add_size]).copy()
            self.types[begin:end] = np.array(types[:first_add_size]).copy()
            self.timestamps[begin:end] = np.array(timestamps[:first_add_size]).copy()
            self.prioritys[begin:end] = self.max_p
            
            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = np.array(obs[first_add_size:]).copy()
            self.next_observations[begin:end] = np.array(next_obs[first_add_size:]).copy()
            self.actions[begin:end] = np.array(actions[first_add_size:]).copy()
            self.rewards[begin:end] = np.array(rewards[first_add_size:]).copy()
            self.terminals[begin:end] = np.array(terminals[first_add_size:]).copy()
            self.types[begin:end] = np.array(types[first_add_size:]).copy()
            self.timestamps[begin:end] = np.array(timestamps[first_add_size:]).copy()
            self.prioritys[begin:end] = self.max_p
            
            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = np.array(obs).copy()
            self.next_observations[begin:end] = np.array(next_obs).copy()
            self.actions[begin:end] = np.array(actions).copy()
            self.rewards[begin:end] = np.array(rewards).copy()
            self.terminals[begin:end] = np.array(terminals).copy()
            self.types[begin:end] = np.array(types).copy()
            self.timestamps[begin:end] = np.array(timestamps).copy()
            self.prioritys[begin:end] = self.max_p

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        p_tensor = torch.tensor(self.prioritys,dtype=torch.float32).to(DEFAULT_DEVICE).squeeze()
        batch_indices = torch.multinomial(p_tensor,batch_size,True).cpu().numpy()
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy(),
            "idx": batch_indices.copy(),
            "types": self.types[batch_indices].copy(),
        }
    
    def update_priority(self, idx, priority):
        self.prioritys[idx] = priority.reshape(-1,1)
        self.max_p = max(self.max_p, priority.max())

    def update_offline_priority(self, priority):
        offline_indices = np.where(self.types == 1)
        self.prioritys[offline_indices] = priority

    def update_timestamps(self):
        indices = np.where(self.timestamps >= 1)
        self.timestamps[indices] += 1

    def update_all_priority(self, max_iters=300000):

        indices = np.where(self.timestamps >= 1)
        self.prioritys[indices] *= np.sqrt(1 - (self.timestamps[indices] - 1) / max_iters)
    
    def sample_online(self, batch_size):
        offline_indices = np.where(self.types == 1)
        p_tensor = torch.tensor(self.prioritys,dtype=torch.float32).to(DEFAULT_DEVICE)
        p_tensor[offline_indices] = 0
        p_tensor = p_tensor.squeeze()
        batch_indices = torch.multinomial(p_tensor,batch_size,True).cpu().numpy()
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy(),
            "idx": batch_indices.copy(),
            "types": self.types[batch_indices].copy(),
        }
    