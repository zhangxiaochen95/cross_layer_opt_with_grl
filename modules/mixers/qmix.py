import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"""

    def __init__(self, state_shape, n_agents, args):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        ts, bs = agent_qs.size(0), agent_qs.size(1)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(ts, bs, 1)
        return q_tot


if __name__ == '__main__':
    from types import SimpleNamespace as SN

    n_agents = 3
    state_shape = 12
    mixing_embed_dim = 32
    config = dict(n_agents=n_agents, state_shape=state_shape, mixing_embed_dim=mixing_embed_dim, hypernet_embed=64)
    args = SN(**config)
    bs = 8
    T = 25
    mixer = QMixer(state_shape, n_agents, args)
    states = th.rand(T, bs, state_shape)
    qs = th.rand(T, bs, n_agents)
    out = mixer(qs, states)
