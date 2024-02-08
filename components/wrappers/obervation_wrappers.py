from abc import abstractmethod
import numpy as np
import torch as th
from torch import Tensor
import dgl
from dgl import DGLGraph

from gym.spaces.utils import flatten_space, flatten
from components.wrappers.wrappers import ObservationWrapper


REGISTRY = {}


class FlatObservation(ObservationWrapper):
    def __init__(self, env):
        super(FlatObservation, self).__init__(env)
        self.observation_space = [flatten_space(self.env.observation_space[i]) for i in range(self.n_agents)]

    def get_obs_size(self):
        return [self.observation_space[i].shape[0] for i in range(self.n_agents)]

    def observation(self, obs):
        flat_obs = [flatten(self.env.observation_space[i], agent_obs) for i, agent_obs in enumerate(obs)]
        return th.as_tensor(np.stack(flat_obs), dtype=th.float32)


REGISTRY['flat'] = FlatObservation


class RelationalObservation(ObservationWrapper):
    def __init__(self, env):
        super(RelationalObservation, self).__init__(env)

    def get_obs_size(self):
        sizes = []
        for obs_space_agent in self.observation_space:
            obs_size = {}
            for k in obs_space_agent:
                obs_size[k] = obs_space_agent[k].shape[-1]
                if k != 'agent':
                    obs_size[k] -= 1  # Drop visibility.
            sizes.append(obs_size)
        return sizes

    def observation(self, obs):
        rel_obs = []
        for agent_obs in obs:

            data_dict = {('agent', 'talks', 'agent'): ([], [])}
            num_nodes_dict = {'agent': 1}
            feat = {'agent': th.as_tensor(agent_obs['agent'], dtype=th.float).unsqueeze(0)}

            for k, v in agent_obs.items():
                if k != 'agent':
                    ent_ids = np.equal(v[:, 0], 1)
                    n_ents = ent_ids.sum()
                    data_dict[(k, 'nearby', 'agent')] = (th.arange(n_ents), th.zeros(n_ents, dtype=th.long))
                    num_nodes_dict[k] = n_ents
                    feat[k] = th.as_tensor(agent_obs[k][ent_ids, 1:], dtype=th.float)

            graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
            graph.ndata['feat'] = feat
            rel_obs.append(graph)

        return dgl.batch(rel_obs)


REGISTRY['rel'] = RelationalObservation


class CommWrapper(ObservationWrapper):
    def __init__(self, env):
        super(CommWrapper, self).__init__(env)

    def get_obs_size(self):
        return self.env.get_obs_size()

    def observation(self, obs):
        u, v = [], []
        if hasattr(self.env, "get_agent_visibility_matrix"):
            agent_vis_mat = self.env.get_agent_visibility_matrix()
        else:
            agent_vis_mat = np.ones((self.n_agents, self.n_agents), dtype=bool)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if agent_vis_mat[i, j]:
                    u.append(i)
                    v.append(j)

        if isinstance(obs, Tensor):
            # comm_graph = dgl.graph((u, v), num_nodes=self.n_agents)
            comm_graph = dgl.heterograph({('agent', 'talks', 'agent'): (u, v)}, num_nodes_dict={'agent': self.n_agents})
            comm_graph.ndata['feat'] = obs

        elif isinstance(obs, DGLGraph):
            data_dict = {c_etype: ([], []) for c_etype in obs.canonical_etypes}
            data_dict.update({('agent', 'talks', 'agent'): (u, v)})
            num_nodes_dict = {ntype: 0 for ntype in obs.ntypes}
            num_nodes_dict.update({'agent': self.n_agents})
            comm_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
            comm_graph = dgl.merge([obs, comm_graph])
        else:
            raise TypeError("Unrecognized type of local observations.")
        return comm_graph


REGISTRY['comm'] = CommWrapper


class GraphObservation(ObservationWrapper):
    """Builds a heterograph as observations of all agents in the env.

    Note that this wrapper does not call `.get_shared_obs()` of the env. Instead, it requires:
    - method `.get_graph_inputs()`
    - property `obs_graph_feats`
    Thus it is less inclusive than `RelationalSharedObservation` (which supports any env providing dict shared obs).
    """

    def __init__(self, env):
        super(GraphObservation, self).__init__(env)
        assert hasattr(self.env, "get_graph_inputs"), "Absence of graph obs callback!"
        assert hasattr(self.env, "graph_feats"), "Absence of graph obs feats!"

    def get_obs(self):
        obs_relations = self.env.get_graph_inputs()
        graph_data = obs_relations['graph_data']  # Define edges
        num_nodes_dict = obs_relations['num_nodes_dict']  # Number of nodes
        node_feats = obs_relations['ndata']  # Node features
        edge_feats = obs_relations.get('edata')  # Edge features

        # Create heterograph as observation.
        obs_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        # Add node/edge features.
        for ntype in obs_graph.ntypes:
            obs_graph.nodes[ntype].data['feat'] = th.as_tensor(node_feats[ntype], dtype=th.float)
        if edge_feats is not None:
            for etype in obs_graph.etypes:
                obs_graph.edges[etype].data['feat'] = th.as_tensor(edge_feats[etype], dtype=th.float)

        # Remove redundant nodes.
        obs_graph = dgl.compact_graphs(obs_graph, always_preserve=dict(agent=th.arange(self.n_agents)),
                                       copy_ndata=True, copy_edata=True)

        return obs_graph

    def get_obs_size(self):
        return [getattr(self.env, "graph_feats")] * self.n_agents


REGISTRY['graph'] = GraphObservation

