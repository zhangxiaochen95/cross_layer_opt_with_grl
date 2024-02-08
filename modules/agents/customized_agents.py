import torch as th
from torch import Tensor
import torch.nn as nn

import dgl
import dgl.nn.pytorch as dglnn
from dgl import DGLGraph
from modules.basics import *
from modules.dueling import DuelingLayer
from modules.encoders import REGISTRY as enc_REGISTRY
from modules.graph_nn import NeighborSelector
from modules.gn_blks import *
from modules.utils import pad_edge_output


class AdHocRelationalController(nn.Module):
    """Graph agent"""

    def __init__(self, obs_shape, act_size, args) -> None:
        super(AdHocRelationalController, self).__init__()
        self._obs_shape = obs_shape  # Shape of observations
        self._act_size = act_size  # Number of discrete actions
        self._obs = args.obs  # Observation format
        self._hidden_size = args.hidden_size  # Hidden size

        self.f_enc = enc_REGISTRY[self._obs](self._obs_shape, self._hidden_size, args)  # Observation encoder
        self.rnn = RnnLayer(self._hidden_size, args)
        self.f_out = NeighborSelector(self._obs_shape['nbr'], self._hidden_size, args.n_pow_opts, 1, self._hidden_size, args)

    def init_hidden(self) -> Tensor:
        """Initializes RNN hidden states."""
        return th.zeros(1, self._hidden_size)

    def forward(self, obs, h):
        # Encode observations.
        x = self.f_enc(obs)
        # Update RNN hidden states independently when communication is disabled.
        x, h = self.rnn(x, h)
        # Compute logits for action selection.
        logits = self.f_out(obs[('nbr', 'nearby', 'agent')], {'agent': x, 'nbr': obs.nodes['nbr'].data['feat']})
        # print(f"logits.size() = {logits.size()}")
        return logits, h


class AdHocGraphController(nn.Module):
    def __init__(self, obs_shape, act_size, args):
        super(AdHocGraphController, self).__init__()

        self._obs_shape = obs_shape  # Shape of observations
        self._act_size = act_size  # Number of discrete actions
        self.args = args

        self._hidden_size = args.hidden_size  # Hidden size
        self._activation = act_REGISTRY[args.activation]  # Callable to instantiate an activation function

        self.khops = args.khops
        if self.khops == 1:
            self.enc = nn.ModuleDict({
                '1hop': NodeGNBlock((self._obs_shape['nbr'], self._obs_shape['agent']),
                                    self._obs_shape['hop'],
                                    self._hidden_size,
                                    activation_type=args.activation),
            })
        elif self.khops == 2:
            self.enc = nn.ModuleDict({
                '2hop': NodeGNBlock((self._obs_shape['nbr'], self._obs_shape['nbr']),
                                    self._obs_shape['hop'],
                                    self._hidden_size,
                                    activation_type=args.activation),
                '1hop': NodeGNBlock((self._hidden_size, self._obs_shape['agent']),
                                    self._obs_shape['hop'],
                                    self._hidden_size,
                                    activation_type=args.activation),
            })

        self.rnn = RnnLayer(self._hidden_size, args)
        inter_nbr_feats = self._hidden_size if self.khops == 2 else self._obs_shape['nbr']
        self.f_out = EdgeGNBlock((inter_nbr_feats, self._hidden_size),
                                 self._obs_shape['hop'],
                                 1,
                                 args.n_pow_opts,
                                 self._hidden_size)

    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """Initializes RNN hidden states."""
        return th.zeros(1, self._hidden_size).expand(batch_size, -1)

    def forward(self, obs, h):
        feat = obs.ndata['feat']

        if self.khops == 2:
            g_2hop = obs['2hop']
            x_nbr = self.enc['2hop'](g_2hop, feat['nbr'], g_2hop.edata['feat'])
        else:
            x_nbr = feat['nbr']

        g_1hop = obs['1hop']
        x = self.enc['1hop'](g_1hop, (x_nbr, feat['agent']), g_1hop.edata['feat'])

        # Update RNN hidden states independently when communication is disabled.
        x, h = self.rnn(x, h)
        # Compute logits for action selection.
        agent_out, nbr_out = self.f_out(g_1hop, (x_nbr, x), g_1hop.edata['feat'])
        # print(f"agent_out.size() = {agent_out.size()}")
        # print(f"nbr_out.size() = {nbr_out.size()}")
        padded_nbr_out = pad_edge_output(g_1hop, nbr_out, self.args.max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        # print(f"q_vals.size() = {q_vals.size()}")
        return q_vals, h
