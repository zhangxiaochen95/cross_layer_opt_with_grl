r"""Miscellaneous stuff used in learning"""

from typing import Union, Optional

import numpy as np
from numpy import ndarray
import torch as th
from torch import Tensor
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import dgl
from dgl import DGLGraph

from gym.spaces.discrete import Discrete


# --- Env ---

def get_random_actions(avail_actions):
    """Randomly draws discrete actions for agents in the env.

    Note that only discrete action space is supported.
    """
    rand_act_list = []
    for avail_agent_actions in avail_actions:
        rand_actions = Categorical(th.tensor(avail_agent_actions, dtype=th.int)).sample().long()
        rand_act_list.append(rand_actions.item())
    return rand_act_list

# --- Shape manipulation ---


def cat(chunks: list[Union[Tensor, DGLGraph]]) -> Union[Tensor, DGLGraph]:
    """Concatenates data held by a list."""

    if isinstance(chunks[0], Tensor):
        return th.cat(chunks)
    elif isinstance(chunks[0], DGLGraph):
        return dgl.batch(chunks)
    else:
        raise TypeError("Unrecognised data type.")


def split(data: Tensor, n_agents: int) -> list[Tensor]:
    """Splits concatenated Tensor."""

    assert data.size(0) % n_agents == 0, "Cannot split data due to inconsistent shape."
    chunks = th.split(data, n_agents)
    return list(chunks)


# --- Discrete actions handling ---


def get_masked_categorical(logits: Tensor, avail_actions: Optional[Tensor] = None) -> Categorical:
    # probs = F.softmax(logits, dim=-1)
    # if avail_actions is not None:
    #     probs[avail_actions == 0] = 0
    # masked_c = Categorical(probs=probs)

    if avail_actions is not None:
        logits[avail_actions == 0] = -1e10
    masked_categorical = Categorical(logits=logits)
    return masked_categorical


def onehot_from_logits(logits: Tensor, avail_logits: Optional[Tensor] = None, eps: float = 0.0):
    """Returns one-hot samples of actions from logits using epsilon-greedy strategy."""
    # Mask unavailable actions.
    # TODO: This operation is in-place, which collapses back-propagation.
    if avail_logits is not None:
        logits[avail_logits == 0] = -1e10
    # Get best actions in one-hot form.
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    else:
        # Get random actions in one-hot form.
        rand_acs = Variable(th.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # Chooses between best and random actions using epsilon greedy
        return th.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(th.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Samples from Gumbel(0, 1)."""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)


def gumbel_softmax_sample(logits, avail_logits, temperature):
    """Draws a sample from the Gumbel-Softmax distribution."""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    # Mask unavailable actions.
    if avail_logits is not None:
        y[avail_logits == 0] = -1e10
    dim = len(logits.shape) - 1
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits: Tensor, avail_logits: Optional[Tensor] = None, temperature: float = 1.0, hard: bool = False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      avail_logits: Mask giving feasibility of actions in logits
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot,
      otherwise it will be a probability distribution that sums to 1 across classes.
    """
    y_soft = gumbel_softmax_sample(logits, avail_logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y_soft)  # Get one-hot sample.
        # Mask has been applied in gumbel-softmax sampling, re-adding mask in discretization causes an error.
        y_out = (y_hard - y_soft).detach() + y_soft
        return y_out
    else:
        return y_soft


def onehot_from_actions(actions: Tensor, n_classes: Union[int, list]):
    # print(f"actions.size() = {actions.size()}")
    if isinstance(n_classes, int):
        onehot_acs = F.one_hot(actions, num_classes=n_classes)
    elif isinstance(n_classes, list):
        acs_per_space = th.split(actions, split_size_or_sections=1, dim=-1)
        onehot_acs_per_space = [F.one_hot(a.squeeze(-1), num_classes=n_classes[s]) for s, a in enumerate(acs_per_space)]
        onehot_acs = th.cat(onehot_acs_per_space, -1)
        # print(f"onehot_acs.size() = {onehot_acs.size()}")
        assert onehot_acs.size(-1) == sum(n_classes), "Incorrect dimension of one-hot actions for multi-discrete space."
    else:
        raise TypeError("Unsupported type of `n_classes` in function `onehot_from_actions`.")
    return onehot_acs


# --- Functions shared by learners ---

def get_clipped_linear_decay(total_steps, threshold):
    assert 1 > threshold >= 0, "Invalid threshold of linear decay."
    return lambda step: max(threshold, 1 - step / total_steps)  # Linear decay and then flat


def mse_loss(error: Tensor, mask: Optional[Tensor] = None):
    """Computes MSE loss from errors."""
    if mask is not None:  # Mask is available.
        # Mask invalid entries.
        mask = mask.expand_as(error)
        masked_error = error * mask
        # print(f"masked_error.squeeze() = \n{masked_error.squeeze()}")
        # Only average valid terms.
        return 0.5 * masked_error.pow(2).sum() / mask.sum()
    else:
        return 0.5 * error.pow(2).mean()


def huber_loss(error: Tensor, mask: Optional[Tensor] = None, delta: float = 10.0):
    """Computes Huber loss from errors.
    See https://en.wikipedia.org/wiki/Huber_loss for more information.
    """
    if mask is not None:
        # Mask invalid entries.
        mask = mask.expand_as(error)
        masked_error = error * mask
        # Determine loss type using delta as threshold.
        l2_rgn = (masked_error.abs() < delta).float()  # Entries using MSE loss
        l1_rgn = 1 - l2_rgn  # Entries using L1 loss
        # Compute loss for each entry.
        loss_per_element = l2_rgn * 0.5 * masked_error.pow(2) + l1_rgn * delta * (masked_error.abs() - 0.5 * delta)
        return loss_per_element.sum() / mask.sum()  # Only average valid terms.

    else:
        l2_rgn = (error.abs() < delta).float()
        l1_rgn = 1 - l2_rgn
        loss_per_element = l2_rgn * 0.5 * error.pow(2) + l1_rgn * delta * (error.abs() - 0.5 * delta)
        return loss_per_element.mean()  # Only average valid terms.


def soft_target_update(policy, target, polyak: float):
    """Smoothly updates target network from learnt policy via polyak averaging."""
    for p, p_targ in zip(policy.parameters(), target.parameters()):
        # In-place operations "mul_", "add_" are used to update target.
        p_targ.data.mul_(polyak)
        p_targ.data.add_((1 - polyak) * p.data)
