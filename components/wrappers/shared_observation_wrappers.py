from abc import abstractmethod
import numpy as np
import torch as th
from torch import Tensor
import dgl
from dgl import DGLGraph

from gym.spaces.utils import flatten_space, flatten
from components.wrappers.wrappers import SharedObservationWrapper


REGISTRY = {}


class FlatSharedObservation(SharedObservationWrapper):
    def __init__(self, env):
        super(FlatSharedObservation, self).__init__(env)
        self.shared_observation_space = [flatten_space(self.env.shared_observation_space[i]) for i in range(self.n_agents)]

    def get_shared_obs_size(self):
        return [self.shared_observation_space[i].shape[0] for i in range(self.n_agents)]

    def shared_observation(self, shared_obs):
        flat_shared_obs = [flatten(self.env.shared_observation_space[i], agent_shared_obs) for i, agent_shared_obs in enumerate(shared_obs)]
        return th.as_tensor(np.stack(flat_shared_obs), dtype=th.float32)


REGISTRY['flat'] = FlatSharedObservation


class RelationalSharedObservation(SharedObservationWrapper):
    def __init__(self, env):
        super(RelationalSharedObservation, self).__init__(env)

    def get_shared_obs_size(self):
        sizes = []
        for shared_obs_space_agent in self.shared_observation_space:
            shared_obs_size = {}
            for k in shared_obs_space_agent:
                shared_obs_size[k] = shared_obs_space_agent[k].shape[-1]
            sizes.append(shared_obs_size)
        return sizes

    def shared_observation(self, shared_obs):
        graph_shared_obs = []
        for agent_shared_obs in shared_obs:

            data_dict = {('agent', 'talks', 'agent'): ([], [])}
            num_nodes_dict = {'agent': 1}
            feat = {'agent': th.as_tensor(agent_shared_obs['agent'], dtype=th.float).unsqueeze(0)}

            for k, v in agent_shared_obs.items():
                if k != 'agent':
                    n_ents = agent_shared_obs[k].shape[0]
                    data_dict[(k, 'nearby', 'agent')] = (th.arange(n_ents), th.zeros(n_ents, dtype=th.long))
                    num_nodes_dict[k] = n_ents
                    feat[k] = th.as_tensor(agent_shared_obs[k], dtype=th.float)

            graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
            graph.ndata['feat'] = feat
            graph_shared_obs.append(graph)

        return dgl.batch(graph_shared_obs)


REGISTRY['rel'] = RelationalSharedObservation


class GraphSharedObservation(SharedObservationWrapper):
    """Builds a heterograph as shared observations of all agents in the env.

    Note that this wrapper does not call `.get_shared_obs()` of the env. Instead, it requires:
    - method `.get_graph_inputs()`
    - property `graph_feats`
    Thus it is less inclusive than `RelationalSharedObservation` (which supports any env providing dict shared obs).
    """

    def __init__(self, env):
        super(GraphSharedObservation, self).__init__(env)
        assert hasattr(self.env, "get_graph_inputs"), "Absence of graph shared obs callback!"
        assert hasattr(self.env, "graph_feats"), "Absence of graph shared obs feats!"

    def get_shared_obs(self):
        shared_obs_relations = self.env.get_graph_inputs()
        graph_data = shared_obs_relations['graph_data']  # Define edges
        num_nodes_dict = shared_obs_relations['num_nodes_dict']  # Number of nodes
        node_feats = shared_obs_relations['ndata']  # Node features
        edge_feats = shared_obs_relations.get('edata')  # Edge features

        # Create heterograph as shared observation.
        shared_obs_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        # Add node/edge features.
        for ntype in shared_obs_graph.ntypes:
            shared_obs_graph.nodes[ntype].data['feat'] = th.as_tensor(node_feats[ntype], dtype=th.float)
        if edge_feats is not None:
            for etype in shared_obs_graph.etypes:
                shared_obs_graph.edges[etype].data['feat'] = th.as_tensor(edge_feats[etype], dtype=th.float)

        # Remove redundant nodes.
        shared_obs_graph = dgl.compact_graphs(shared_obs_graph, always_preserve=dict(agent=th.arange(self.n_agents)),
                                              copy_ndata=True, copy_edata=True)
        # print(shared_obs_graph)
        # print(shared_obs_graph.ndata['feat'])
        # print(shared_obs_graph.edata['feat'])
        return shared_obs_graph

    def get_shared_obs_size(self):
        return [getattr(self.env, "graph_feats")] * self.n_agents


REGISTRY['graph'] = GraphSharedObservation


if __name__ == '__main__':
    from envs.ad_hoc.ad_hoc import AdHocEnv
    env = AdHocEnv('2flows')
    env.reset()
    shared_obs = env.get_shared_obs()
    env = GraphSharedObservation(env)
    shared_obs = env.get_shared_obs()