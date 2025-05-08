import numpy as np


class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        singel_sample = False
        if len(obs.shape) == 1:
            obs = obs.reshape(1,-1)
            singel_sample = True
        if len(act.shape) == 1:
            act = act.reshape(1,-1)
        if len(next_obs.shape) == 1:
            next_obs = next_obs.reshape(1,-1)
        
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        assert isinstance(obs, np.ndarray)
            
        # Status: landing and flight
        done = (np.abs(next_obs[..., 1]) <= 0.05)
            
        if singel_sample:
            done = done[0]
            done = done.item()

        return done