import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected MLP that outputs a scalar HJ value."""

    def __init__(self, input_dim: int, hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect x shape (batch, input_dim)
        out = self.net(x)
        return out.squeeze(-1)


def build_mlp(input_dim: int, hidden_dims=(256, 256), activation=nn.ReLU):
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
