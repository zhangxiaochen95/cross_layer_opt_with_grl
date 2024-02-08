import torch as th
import torch.nn as nn

from modules.activations import REGISTRY as act_REGISTRY


class DuelingLayer(nn.Module):
    """Dueling architecture separating state values and advantage functions

    For more details, refer to "Dueling Network Architectures for Deep Reinforcement Learning".
    """
    def __init__(self, hidden_size, n_actions, args):
        super(DuelingLayer, self).__init__()
        self._hidden_size = hidden_size
        self._activation = act_REGISTRY[args.activation]  # Activation function

        self.f_val = [nn.Linear(self._hidden_size, self._hidden_size)]
        self.f_adv = [nn.Linear(self._hidden_size, self._hidden_size)]
        if args.use_layer_norm:  # Layer normalization is used.
            self.f_val += [nn.LayerNorm(self._hidden_size)]
            self.f_adv += [nn.LayerNorm(self._hidden_size)]
        self.f_val += [nn.ReLU(), nn.Linear(self._hidden_size, 1)]
        self.f_adv += [nn.ReLU(), nn.Linear(self._hidden_size, n_actions)]

        self.f_val = nn.Sequential(*self.f_val)
        self.f_adv = nn.Sequential(*self.f_adv)

    def forward(self, x):
        vals = self.f_val(x)
        advs = self.f_adv(x)
        # print(f"vals.size() = {vals.size()}")
        # print(f"advs.size() = {advs.size()}")
        # print(f"advs.mean(-1) = {advs.mean(-1)}")
        # print(f"advs.mean(-1, keepdim=True).expand_as(a) = \n{advs.mean(-1, keepdim=True).expand_as(advs)}")
        return vals + advs - advs.mean(-1, keepdim=True)


if __name__ == '__main__':
    from types import SimpleNamespace as SN
    args = SN(**dict(activation='relu', use_layer_norm='False'))
    hidden_size = 32
    n_actions = 5
    dueling = DuelingLayer(hidden_size, n_actions, args)

    batch_size = 10
    x = th.rand(batch_size, hidden_size)
    y = dueling(x)
    print(f"y.size() = {y.size()}")
    print(getattr(args, "name", 0))
