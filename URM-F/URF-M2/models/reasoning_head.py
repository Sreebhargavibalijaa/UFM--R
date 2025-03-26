import torch
import torch.nn as nn
from .tabular_encoder import NAM

class UFM2ReasoningHead(nn.Module):
    def __init__(self, tabular_dim, text_dim):
        super().__init__()
        self.nam = NAM(tabular_dim)
        self.text_proj = nn.Linear(text_dim, 1)
        self.fc = nn.Linear(2, 1)

    def forward(self, tabular_input, text_embed):
        tab_score, tab_contribs = self.nam(tabular_input)
        text_score = self.text_proj(text_embed)
        combined = torch.cat([tab_score, text_score], dim=1)
        return self.fc(combined), tab_contribs
