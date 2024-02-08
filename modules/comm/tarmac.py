import torch as th
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class TarMAC(nn.Module):
    """TarMAC: Targeted Multi-Agent Communication"""

    def __init__(self, hidden_size, args):
        super(TarMAC, self).__init__()
        self._hidden_size = hidden_size  # Size of hidden states
        self._msg_size = args.msg_size  # Size of messages
        self._key_size = args.key_size  # Size of signatures and queries
        self._n_rounds = args.n_rounds  # Number of multi-round communication

        self.f_val = nn.Linear(2 * self._hidden_size, self._msg_size)  # Value function (producing messages)
        self.f_sign = nn.Linear(2 * self._hidden_size, self._key_size)  # Signature function (predicting keys at Tx)
        self.f_que = nn.Linear(2 * self._hidden_size, self._key_size)  # Query function (predicting keys at Rx)
        self.f_udt = nn.GRUCell(self._hidden_size + self._msg_size, self._hidden_size)  # RNN update function

    def forward(self, g: DGLGraph, feat: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        with g.local_scope():
            g.ndata['x'] = feat
            for l in range(self._n_rounds):
                g.ndata['h'] = h  # Get the latest hidden states.

                # Build inputs to modules for communication.
                inputs = th.cat((g.srcdata['x'], g.srcdata['h'].detach()), 1)
                # Get signatures, values at source nodes.
                g.srcdata.update(dict(v=self.f_val(inputs), s=self.f_sign(inputs)))
                # Predict queries at destination nodes.
                g.dstdata.update(dict(q=self.f_que(inputs)))

                # Get attention scores on each edge by Dot-product of signature and query.
                g.apply_edges(fn.u_dot_v('s', 'q', 'e'))
                # Normalize attention scores
                e = g.edata.pop('e') / self._key_size  # Divide by key-size
                g.edata['a'] = edge_softmax(g, e)
                # Aggregated messages by weighted sum
                g.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'c'))

                # Update the hidden states of GRU.
                h = self.f_udt(th.cat((g.ndata['x'], g.ndata['c']), 1), h)
            return h, h
