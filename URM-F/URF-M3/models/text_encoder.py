import torch.nn as nn
import torch

class InterpretableAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        return self.softmax(scores)[:, 0, :]
