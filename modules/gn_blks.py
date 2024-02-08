r"""Graph network (GN) blocks"""

from typing import Union

import torch as th
import torch.nn as nn

import dgl
from dgl.base import DGLError, DGLWarning
from dgl.utils import expand_as_pair
import dgl.function as fn
from modules.activations import REGISTRY as act_REGISTRY


class NodeGNBlock(nn.Module):
    """Node-focused graph network (GN) block

    Components and update rules follow the paper
    "Relational inductive biases, deep learning, and graph networks" (https://arxiv.org/abs/1806.01261v1).
    """
    def __init__(self,
                 in_node_feats: Union[int, tuple[int, int]],  # Size of input node features
                 in_edge_feats: int,  # Size of input edge features
                 out_node_feats: int,  # Size of output node features
                 aggregator_type: str = 'mean',
                 activation_type: str = 'relu',
                 ):
        super(NodeGNBlock, self).__init__()

        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._out_node_feats = out_node_feats

        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        self._aggr_type = aggregator_type
        self.activation = act_REGISTRY[activation_type]

        # Edge update function.
        self.f_e = nn.Sequential(
            nn.Linear(self._in_src_node_feats + self._in_edge_feats + self._in_dst_node_feats, self._out_node_feats),
            self.activation()
        )
        # Node update function.
        self.f_v = nn.Sequential(
            nn.Linear(self._out_node_feats + self._in_dst_node_feats, self._out_node_feats),
            self.activation(),
        )

    def edge_update_func(self, edges):
        x = th.cat([edges.src['v_i'], edges.data['e'], edges.dst['v_j']], 1)
        return {'m': self.f_e(x)}

    def forward(self, graph, node_feats, edge_feats):
        _reducer = getattr(fn, self._aggr_type)
        with graph.local_scope():
            if isinstance(node_feats, tuple):
                src_node_feats, dst_node_feats = node_feats
            else:
                src_node_feats = dst_node_feats = node_feats

            graph.srcdata.update({'v_i': src_node_feats})
            graph.dstdata.update({'v_j': dst_node_feats})
            graph.edata.update({'e': edge_feats})

            graph.update_all(self.edge_update_func, _reducer('m', 'neigh'))
            return self.f_v(th.cat([graph.dstdata['neigh'], graph.dstdata['v_j']], 1))


class EdgeGNBlock(nn.Module):
    def __init__(self,
                 in_node_feats: Union[int, tuple[int, int]],  # Size of input node features
                 in_edge_feats: int,  # Size of input edge features
                 out_node_feats: int,  # Size of output node features
                 out_edge_feats: int,  # Size of output edge features
                 hidden_size: int,  # Hidden size
                 activation_type: str = 'relu',  # Activation function
                 ):
        super(EdgeGNBlock, self).__init__()

        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self._hidden_size = hidden_size

        self.activation = act_REGISTRY[activation_type]
        # Edge update function.
        self.f_e = nn.Sequential(
            nn.Linear(self._in_src_node_feats + self._in_edge_feats + self._in_dst_node_feats, self._hidden_size),
            self.activation(),
            nn.Linear(self._hidden_size, self._out_edge_feats)
        )

        self.f_v = nn.Linear(self._in_dst_node_feats, self._out_node_feats)

    def edge_update_func(self, edges):
        x = th.cat([edges.src['v_i'], edges.data['e'], edges.dst['v_j']], 1)
        return {'e2': self.f_e(x)}

    def node_update_func(self, nodes):
        return {'v2_j': self.f_v(nodes.data['v_j'])}

    def forward(self, graph, node_feats, edge_feats):
        assert graph.is_unibipartite, "Only uni-bipartite graph is supported by `EdgeGNBlock`."
        with graph.local_scope():
            if isinstance(node_feats, tuple):
                src_node_feats, dst_node_feats = node_feats
            else:
                src_node_feats = dst_node_feats = node_feats

            graph.srcdata.update({'v_i': src_node_feats})
            graph.dstdata.update({'v_j': dst_node_feats})
            graph.edata.update({'e': edge_feats})

            graph.apply_edges(self.edge_update_func)
            v2_j = self.f_v(dst_node_feats)
            return v2_j, graph.edata['e2']


if __name__ == '__main__':
    g = dgl.heterograph({('nbr', 'nearby', 'agent'): (th.tensor([0, 1, 2, 1, 2, 3]), th.tensor([0, 0, 0, 1, 1, 1]))})
    print(g[('nbr', 'nearby', 'agent')])
    in_node_feats = (8, 7)
    in_edge_feats = 6
    out_node_feats = 5
    out_edge_feats = 4

    gnn = EdgeGNBlock(in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, 32)
    node_feats = (th.rand(4, in_node_feats[0]), th.rand(2, in_node_feats[1]))
    edge_feats = th.rand(6, in_edge_feats)
    node_feats, edge_feats = gnn(g, node_feats, edge_feats)
    print(f"node_feats.size() = {node_feats}, \nedge_feats.size() = {edge_feats}")