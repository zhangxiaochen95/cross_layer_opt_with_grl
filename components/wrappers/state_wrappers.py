import torch as th
from gym.spaces.utils import flatten_space, flatten
from envs.multi_agent_env import MultiAgentEnv
from components.wrappers.wrappers import StateWrapper


REGISTRY = {}


class FlatStateWrapper(StateWrapper):
    def __init__(self, env: MultiAgentEnv):
        super(FlatStateWrapper, self).__init__(env)
        self.state_space = flatten_space(self.env.state_space)

    def state(self, stat):
        stat = flatten(self.env.state_space, stat)
        stat = th.as_tensor(stat, dtype=th.float)  # Covert to Tensor
        return th.atleast_2d(stat)  # Check dimension

    def get_state_size(self):
        return self.state_space.shape[0]


REGISTRY['flat'] = FlatStateWrapper
