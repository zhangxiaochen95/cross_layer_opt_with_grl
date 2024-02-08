from typing import Union
import torch as th
import torch.nn as nn
import dgl

from modules.activations import REGISTRY as act_REGISTRY


class NeighborSelector(nn.Module):
    def __init__(self, nbr_in_feats, agent_in_feats, nbr_out_feats, agent_out_feats, hidden_size, args):
        super(NeighborSelector, self).__init__()

        self.nbr_in_feats = nbr_in_feats
        self.agent_in_feats = agent_in_feats
        self.nbr_out_feats = nbr_out_feats
        self.agent_out_feats = agent_out_feats

        self.device = args.device
        self.max_nbrs = args.max_nbrs

        self._hidden_size = hidden_size
        self._activation = act_REGISTRY[args.activation]  # Activation function
        self.nbr_predictor = nn.Sequential(
            nn.Linear(nbr_in_feats + agent_in_feats, self._hidden_size),
            self._activation(),
            nn.Linear(self._hidden_size, nbr_out_feats)
        )
        self.agent_predictor = nn.Linear(self._hidden_size, agent_out_feats)

    def predict_entity_score(self, edges):
        h_ent = edges.src['x']
        h_agent = edges.dst['x']
        # print(f"h_ent.size() = {h_ent.size()}, h_agent.size() = {h_agent.size()}")
        score = self.nbr_predictor(th.cat([h_ent, h_agent], 1))
        return {'score': score}  # (batch_size * n_agents, nbr_out_feats)

    def forward(self, graph, x):
        with graph.local_scope():
            # Get scores of neighbors and own with graph convolution.
            graph.ndata['x'] = x
            graph.apply_edges(self.predict_entity_score)
            nbr_scores = graph.edata['score']  # shape (batch_size * n_agents * nbr_per_agent, 1)
            own_score = self.agent_predictor(x['agent'])  # shape (batch_size * n_agents, 1)
            # print(f"nbr_scores.size() = {nbr_scores.size()}, own_score.size() = {own_score.size()}")

            nbrs_per_agent = graph.in_degrees().tolist()  # Number of neighbors around each agent
            # Split nbr scores for each agent and pad zero to `max_nbr`.
            nbr_scores = th.split(nbr_scores, split_size_or_sections=nbrs_per_agent, dim=0)  # Each entry has shape (n_nbrs, nbr_out_feats)
            pad_zeros = [th.zeros(self.nbr_out_feats * (self.max_nbrs - nbrs_per_agent[i]), dtype=th.float, device=self.device) for i in range(len(nbrs_per_agent))]  # all-zero with shape (max_nbrs - n_nbrs) of each agent
            # print(f"nbr_scores.size() = {nbr_scores}")
            # print(f"self.nbr_out_feats * nbrs_per_agent = {self.nbr_out_feats * nbrs_per_agent}")
            # print(f"pad_zeros = \n{pad_zeros}")
            # print(f"nbr_scores = \n{nbr_scores}")
            # Pad nbr scores of each agent with all-zero vector.
            padded_nbr_scores = []
            for agent_idx, score in enumerate(nbr_scores):
                padded_nbr_scores.append(th.cat((score.flatten(), pad_zeros[agent_idx])))
            padded_nbr_scores = th.stack(padded_nbr_scores)  # Shape (batch_size * n_agents, max_nbrs)
            all_scores = th.cat([padded_nbr_scores, own_score], 1)
            # print(f"padded_nbr_scores has size {padded_nbr_scores.size()} = \n{padded_nbr_scores}")
            # print(f"all_scores.size() = {all_scores.size()}")
            return all_scores
