import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph


class DiscreteCommunication(nn.Module):
    """Discrete Communication"""

    def __init__(self, hidden_size, args) -> None:
        super(DiscreteCommunication, self).__init__()

        self._hidden_size = hidden_size  # Size of hidden states
        self._msg_size = args.msg_size  # Size of messages
        # Note: In discrete communication, we use 2 digits to denote 1 bit as either (0, 1) or (1, 0).
        # Therefore, outputs from message encoder take twice the number of digits of continuous counterparts.

        self.f_enc = nn.Linear(self._hidden_size + self._hidden_size, 2 * self._msg_size)  # Message function
        self.f_dec = nn.Linear(2 * self._msg_size, 2 * self._msg_size)  # Decoder of aggregated messages
        self.f_udt = nn.GRUCell(self._hidden_size + 2 * self._msg_size, self._hidden_size)  # RNN unit

    def msg_func(self, edges):
        """Encodes discrete messages from local inputs and detached hidden states."""
        # Get logits from message function.
        logits = self.f_enc(th.cat((edges.src['x'], edges.src['h']), 1))
        # When discrete communication is required,
        # we use Gumbel-Softmax function to sample binary messages while keeping gradients for backpropagation.
        disc_msg = F.gumbel_softmax(logits.view(-1, self._msg_size, 2), tau=0.5, hard=True)
        return dict(m=disc_msg.flatten(1))

    def aggr_func(self, nodes):
        """Aggregates incoming discrete messages by element-wise OR operation."""
        aggr_msg = nodes.mailbox['m'].max(1)[0]
        return dict(c=aggr_msg)

    def forward(self, g: DGLGraph, feat: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        with g.local_scope():
            g.ndata['x'], g.ndata['h'] = feat, h.detach()  # Get inputs and the latest hidden states.

            if g.number_of_edges() == 0:
                # When no edge is created, paddle zeros.
                g.dstdata['c'] = th.zeros(feat.shape[0], 2 * self._msg_size)
            else:
                # Otherwise, call message passing between nodes.
                g.update_all(self.msg_func, self.aggr_func)

            # Update the hidden states using inputs, aggregated messages and hidden states.
            h = self.f_udt(th.cat((g.ndata['x'], self.f_dec(g.ndata['c'])), 1), h)
            return h, h
