import torch as th
from torch import Tensor
import torch.nn as nn

from dgl import DGLGraph
from modules.basics import *
from modules.dueling import DuelingLayer
from modules.encoders import REGISTRY as enc_REGISTRY
from modules.comm import REGISTRY as comm_REGISTRY


class CommunicativeAgent(nn.Module):
    """Communicative agent"""

    def __init__(self, obs_shape, act_size, args) -> None:
        super(CommunicativeAgent, self).__init__()
        self._obs_shape = obs_shape  # Shape of observations
        self._act_size = act_size  # Number of discrete actions
        self._obs = args.obs  # Observation format
        self._hidden_size = args.hidden_size  # Hidden size

        self._comm = args.comm  # Communication protocol
        assert self._comm in comm_REGISTRY, "Unrecognised communication protocol."

        self.f_enc = enc_REGISTRY[self._obs](self._obs_shape, self._hidden_size, args)  # Observation encoder
        self.f_comm = comm_REGISTRY[self._comm](self._hidden_size, args)  # Modules performing multi-agent communication
        if getattr(args, "use_dueling", False):
            self.f_out = DuelingLayer(self._hidden_size, self._act_size, args)
        else:
            self.f_out = nn.Linear(self._hidden_size, self._act_size)  # Output layer

    def init_hidden(self) -> Tensor:
        """Initializes RNN hidden states."""
        return th.zeros(1, self._hidden_size)

    def forward(self, obs, h):
        # Encode observations.
        x = self.f_enc(obs)
        # Apply multi-agent communication and update RNN hidden states after communicating among models.
        x, h = self.f_comm(obs['talks'], x, h)  # Note that call subgraph by edge name may cause ambiguity.
        # Compute logits for action selection.
        logits = self.f_out(x)
        return logits, h


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    args = SN(**dict(hidden_size=128))
    print(type(args))
