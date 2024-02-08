import torch as th
from torch import Tensor
import torch.nn as nn

from modules.basics import *
from modules.dueling import DuelingLayer
from modules.encoders import REGISTRY as enc_REGISTRY


class RecurrentAgent(nn.Module):
    """Recurrent agent"""

    def __init__(self, obs_shape, act_size, args) -> None:
        super(RecurrentAgent, self).__init__()
        self._obs_shape = obs_shape  # Shape of observations
        self._act_size = act_size  # Number of discrete actions
        self._obs = args.obs  # Observation form
        self._hidden_size = args.hidden_size  # Hidden size

        self.f_enc = enc_REGISTRY[self._obs](self._obs_shape, self._hidden_size, args)  # Observation encoder
        self.rnn = RnnLayer(self._hidden_size, args)
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
        # Update RNN hidden states independently when communication is disabled.
        x, h = self.rnn(x, h)
        # Compute logits for action selection.
        logits = self.f_out(x)
        return logits, h


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    args = SN(**dict(hidden_size=128))
    print(type(args))
