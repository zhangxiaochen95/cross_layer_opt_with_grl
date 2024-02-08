from typing import Optional
from abc import abstractmethod

import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, OneHotCategorical
import torch.nn.functional as F

REGISTRY = {}


class BaseActionSelector:
    """Base class of action selector"""
    
    def __init__(self, args):
        pass

    @ abstractmethod
    def select_actions(self, logits: Tensor, avail_logits: Optional[Tensor], t: Optional[int], mode: str):
        """Selects actions from logits."""
        raise NotImplementedError


class EpsilonGreedyActionSelector(BaseActionSelector):
    """Epsilon-greedy action selector to balance between exploration and exploitation

    Multi-discrete action space is accepted.
    """

    def __init__(self, args):
        super(EpsilonGreedyActionSelector, self).__init__(args)
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_anneal_time = args.eps_anneal_time

        self.is_multi_discrete = args.is_multi_discrete
        if self.is_multi_discrete:
            self.nvec = args.nvec

    def schedule(self, t: int) -> float:
        return max(self.eps_end, -(self.eps_start - self.eps_end) / self.eps_anneal_time * t + self.eps_start)

    def select_actions(self, logits: Tensor, avail_logits: Optional[Tensor] = None, t: Optional[int] = None,
                       mode: str = 'explore'):
        # Fill available logits if not provided.
        if avail_logits is None:
            avail_logits = th.ones_like(logits)
        # Mask unavailable actions
        masked_logits = logits.clone()
        masked_logits[avail_logits == 0] = -float("inf")
        # Choose actions following the epsilon-greedy strategy.

        if not self.is_multi_discrete:
            greedy_actions = th.argmax(masked_logits, 1)  # Greedy actions
            rand_actions = Categorical(avail_logits).sample().long()  # Random samples from available actions

        else:
            # Split inputs into spaces.
            masked_logits_per_space = th.split(masked_logits, split_size_or_sections=self.nvec, dim=-1)
            avail_logits_per_space = th.split(avail_logits, split_size_or_sections=self.nvec, dim=-1)
            # Greedily select action for each action space.
            greedy_actions = th.stack([th.argmax(chunk, 1) for chunk in masked_logits_per_space]).T
            # Randomly select from available actions for each action space.
            rand_actions = th.stack([Categorical(chunk).sample().long() for chunk in avail_logits_per_space]).T

        # Get action according to mode.
        if mode == 'rand':  # Randomly actions from available logits
            return rand_actions
        elif mode == 'test':  # Greedy actions
            return greedy_actions
        elif mode == 'explore':
            eps_thres = self.schedule(t)  # epsilon threshold
            # Determine whether to pick randon actions or greedy ones.
            pick_rand = (th.rand_like(logits[..., 0]) < eps_thres).long()
            if self.is_multi_discrete:  # Multiple spaces requires an additional dimension.
                pick_rand = th.unsqueeze(pick_rand, -1)
            # Pick either random or greedy actions.
            picked_actions = pick_rand * rand_actions + (1 - pick_rand) * greedy_actions
            return picked_actions
        else:
            raise KeyError("Invalid mode of action selection.")


REGISTRY['epsilon_greedy'] = EpsilonGreedyActionSelector


class CategoricalActionSelector(BaseActionSelector):
    """Categorical action selector for stochastic discrete policies"""
    
    def select_actions(self, logits: Tensor, avail_logits: Optional[Tensor] = None, t: Optional[int] = None,
                       mode: str = 'explore'):
        # Fill available logits if not provided.
        if avail_logits is None:
            avail_logits = th.ones_like(logits)

        # Mask unavailable actions.
        masked_logits = logits.clone()
        masked_logits[avail_logits == 0] = -float("inf")

        # Get action according to mode.
        if mode == 'rand':
            rand_actions = Categorical(avail_logits).sample().long()  # Random samples from available actions
            return rand_actions
        elif mode == 'explore':
            probs = F.softmax(masked_logits, dim=-1)  # Convert logits to probs.
            sampled_actions = Categorical(probs=probs).sample()  # Get action samples from categorical distributions.
            return sampled_actions.unsqueeze(-1)
        elif mode == 'test':
            return th.argmax(masked_logits, dim=-1, keepdim=True)  # Greedy actions
        else:
            raise KeyError("Invalid mode of action selection.")


REGISTRY['categorical'] = CategoricalActionSelector


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    import numpy as np
    import torch.nn.functional as F

    # args = SN(**dict(eps_start=1, eps_end=0.05, anneal_time=50000, temperature=0.1))
    # logits = th.rand(3, 5)
    # avail_actions = th.randint(0, 2, logits.size())
    # avail_actions[:, 3:] = 0
    #
    # action_selector = GumbelSoftmaxMultinomialActionSelector(args)
    # actions = action_selector.select_actions(logits, 10000, avail_actions, test_mode=True, explore=False)
    # print(actions)

    # a = 0.1 * th.tensor([0.5, 0.2, 0.3])
    # a[0] = 0
    # a = th.rand(3, 4)
    # print(a)
    # print(th.argmax(a, dim=1))
    # #
    # m = Categorical(a)
    # actions = m.sample()
    # print(actions)
    # print(actions.view(3, -1))
    # onehot_acs = F.one_hot(actions, num_classes=a.shape[-1])
    # print(onehot_acs)
    nvec = np.array([3, 5])
    args = SN(**dict(eps_start=1, eps_end=0.05, eps_anneal_time=50000), nvec=nvec.tolist(), is_multi_discrete=False)
    action_selector = EpsilonGreedyActionSelector(args)

    batch_size = 10
    logits = th.rand(batch_size, nvec.sum())
    acts = action_selector.select_actions(logits=logits, t=1000)



