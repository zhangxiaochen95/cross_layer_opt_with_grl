import numpy as np
from components.wrappers.wrappers import RewardWrapper
from components.normalizers import ZFilter

REGISTRY = {}


class RewardNormalizer(RewardWrapper):
    def __init__(self, env, args):
        super(RewardNormalizer, self).__init__(env)
        reward_shape = self.n_agents
        self.reward_normalizer = ZFilter(reward_shape)

        self.clip_range = args.max_reward_limit

    def reward(self, rewards):
        rewards = self.reward_normalizer(rewards)
        if self.clip_range is not None:
            rewards = np.clip(rewards, -self.clip_range, self.clip_range)
        return rewards


REGISTRY['norm'] = RewardNormalizer
