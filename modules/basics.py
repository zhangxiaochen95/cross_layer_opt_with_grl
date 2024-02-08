import torch as th
import torch.nn as nn


class RnnLayer(nn.Module):
    """Recurrent layer enabling layer normalization"""

    def __init__(self, hidden_size, args):
        super(RnnLayer, self).__init__()
        self._hidden_size = hidden_size
        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)

        self._use_layer_norm = args.use_layer_norm
        if self._use_layer_norm:
            self.norm = nn.LayerNorm(self._hidden_size)

    def forward(self, x, h):
        h = self.rnn(x, h)
        if self._use_layer_norm:
            y = self.norm(h)
        else:
            y = h
        return y, h
