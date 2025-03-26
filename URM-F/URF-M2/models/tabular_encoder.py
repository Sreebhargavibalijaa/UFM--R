import torch
import torch.nn as nn

class FeatureNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

class NAM(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.feature_nets = nn.ModuleList([FeatureNet(hidden) for _ in range(input_dim)])

    def forward(self, x):
        contribs = [net(x[:, i].unsqueeze(1)) for i, net in enumerate(self.feature_nets)]
        contribs = torch.cat(contribs, dim=1)
        return contribs.sum(dim=1, keepdim=True), contribs
