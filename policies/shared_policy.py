from typing import Optional
import torch as th
from torch import Tensor
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


class SharedPolicy:
    """Policy shared by all homogeneous agents"""

    def __init__(self, env_info, args) -> None:
        self.n_agents = env_info.get('n_agents')

        obs_shape = env_info['obs_shape'][0]  # Shape of observations
        self._build_agents(obs_shape, args.act_size, args)  # Agents
        self.action_selector = action_REGISTRY[args.action_selector](args)  # Action selector

    def _build_agents(self, obs_shape, act_size, args) -> None:
        self.model = agent_REGISTRY[args.agent](obs_shape, act_size, args)

    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """Initializes RNN hidden states of a batch of multi-models."""
        return self.model.init_hidden().expand(batch_size * self.n_agents, -1)

    def forward(self, obs, h: Tensor):
        logits, h = self.model(obs, h)
        return logits, h

    @ th.no_grad()
    def act(self, obs, h: Tensor, avail_actions: Optional[Tensor] = None, t: Optional[int] = None,
            mode: str = 'explore', **kwargs):
        """Selects actions of models from observations."""
        logits, next_h = self.forward(obs, h)
        actions = self.action_selector.select_actions(logits, avail_actions, t, mode)
        return actions, next_h

    def parameters(self):
        """Returns parameters of neural network."""
        return self.model.parameters()

    def to(self, device) -> None:
        """Moves neural network to device."""
        self.model.to(device)

    def load_state(self, other_policy) -> None:
        """Loads the parameters from another policy."""
        self.model.load_state_dict(other_policy.model.state_dict())

    def eval(self) -> None:
        """Sets agent to eval mode"""
        self.model.eval()

    def train(self) -> None:
        """Sets agent to training mode."""
        self.model.train()

    def __repr__(self):
        return f"Shared policy: \n{self.model}"
