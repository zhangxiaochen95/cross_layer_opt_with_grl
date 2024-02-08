from typing import Union
import torch as th
from torch import Tensor
import torch.nn as nn
from dgl import DGLGraph

from modules.activations import REGISTRY as act_REGISTRY


class FlatEncoder(nn.Module):
    """Flat input encoder"""

    def __init__(self, input_shape: int, hidden_size: int, args):
        super(FlatEncoder, self).__init__()

        self._hidden_size = hidden_size  # Hidden size
        self._n_layers = args.n_layers  # Number of fully-connected layers
        self._activation = act_REGISTRY[args.activation]  # Activation function
        self._use_feat_norm = args.use_feat_norm
        self._use_layer_norm = args.use_layer_norm

        # Define input layer.
        if self._use_feat_norm:  # Feature normalization is used.
            self.feat_norm = nn.LayerNorm(input_shape)
        layers = [nn.Linear(input_shape, self._hidden_size)]
        if self._use_layer_norm:  # Add layer normalization.
            layers += [nn.LayerNorm(self._hidden_size)]
        layers += [self._activation()]
        # Define hidden layers.
        for l in range(1, self._n_layers):
            layers += [nn.Linear(self._hidden_size, self._hidden_size)]
            if self._use_layer_norm:  # Add layer normalization.
                layers += [nn.LayerNorm(self._hidden_size)]
            layers += [self._activation()]
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Union[Tensor, DGLGraph]) -> Tensor:
        # Extract feature observation is a graph.
        if isinstance(inputs, DGLGraph):
            inputs = inputs.ndata['feat']
        if self._use_feat_norm:
            inputs = self.feat_norm(inputs)
        return self.layers(inputs)


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    import torch as th
    import torch.nn as nn

    args = SN(**dict(hidden_size=16, n_layers=2, batch_size=10, n_heads=4, activation=nn.ELU))

    obs_shape = 5
    enc = FlatEncoder(obs_shape, args)
    print(enc)
    obs = th.rand(args.batch_size, obs_shape)
    rst = enc(obs)
