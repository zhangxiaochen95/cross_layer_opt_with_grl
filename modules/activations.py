import torch.nn as nn

REGISTRY = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}
