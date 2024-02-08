from collections.abc import Mapping
import torch as th
from torch import Tensor
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
from dgl import DGLHeteroGraph

from modules.activations import REGISTRY as act_REGISTRY


class RelationalEncoder(nn.Module):
    """Relational input encoder
    Observations are heterogeneous graphs of which edges are established from observed entities to models.
    """

    def __init__(self, in_feats_size_dict: Mapping[str, int], hidden_size, args) -> None:
        super(RelationalEncoder, self).__init__()

        assert 'agent' in in_feats_size_dict, "agent features must be reserved in observations."
        in_feats_size_dict_ = in_feats_size_dict.copy()
        agent_feats_size = in_feats_size_dict_.pop('agent')  # Size of input features of models
        self._ntypes = tuple(in_feats_size_dict_.keys())  # Number of entity types in observations (except agent)

        self._hidden_size = hidden_size  # Hidden size
        self._n_heads = args.n_heads  # Number of attention heads
        feats_per_head = self._hidden_size // self._n_heads  # Size of output features per head
        self._activation = act_REGISTRY[args.activation]  # Callable to instantiate an activation function

        # Define a separate module to process each type of entities.
        mods = dict()
        for ntype, feats_size in in_feats_size_dict_.items():
            mods[ntype] = dglnn.GATv2Conv((feats_size, agent_feats_size), feats_per_head, self._n_heads,
                                          allow_zero_in_degree=True, residual=True, activation=self._activation())
        self.f_conv = nn.ModuleDict(mods)  # Dict holding graph convolution layers
        self.f_aggr = nn.Sequential(nn.Linear(self._hidden_size * len(self._ntypes), self._hidden_size),
                                    self._activation())  # MLP aggregator

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        feat = g.ndata['feat']
        outputs = {}
        # Go through all types of entities.
        for stype, etype, dtype in g.canonical_etypes:
            # When an entity type is not specified by modules, skip it.
            if (stype not in self.f_conv) or (etype != 'nearby') or (dtype != 'agent'):
                continue
            # Extract subgraph and apply graph convolution.
            rel_g = g[stype, etype, dtype]
            outputs[stype] = self.f_conv[stype](rel_g, (feat[stype], feat['agent']))
        # Aggregate outputs from different relations to obtain final results.
        rsts = self._aggr_func(outputs)
        return rsts

    def _aggr_func(self, outputs: Mapping[str, Tensor]) -> Tensor:
        """Aggregates outputs from multiple relations.
        An MLP aggregator is used to transform stacked outputs into expected shape.
        """
        # Stack outputs from relations in order.
        stacked = th.stack([outputs[ntype] for ntype in self._ntypes], dim=1)
        # Flatten stacked outputs and pass them through MLP aggregator.
        return self.f_aggr(stacked.flatten(1))


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    import torch as th
    import torch.nn as nn
    import dgl

    args = SN(**dict(hidden_size=16, n_layers=2, batch_size=10, n_heads=4, activation=nn.ELU))

    obs_feats_size_dict = {'gt': 3, 'uav': 2, 'agent': 3}
    enc = RelationalEncoder(obs_feats_size_dict, args)
    print(enc)
    graph_data = {
        ('gt', 'nearby', 'agent'): ((0, 1, 0), (0, 0, 1)),
        ('uav', 'nearby', 'agent'): ((0, 1), (1, 0)),
        ('agent', 'talks', 'agent'): ((0, 1), (1, 0)),
    }
    num_nodes_dict = {'gt': 2, 'uav': 2, 'agent': 2}
    obs_g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    print(obs_g)
    feat = {ntype: th.rand(num_nodes_dict[ntype], obs_feats_size_dict[ntype]) for ntype in obs_feats_size_dict}
    rsts = enc(obs_g, feat)
    print(rsts)
