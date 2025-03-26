import torch.nn as nn
from .tabular_encoder import NAM
import torch

class NAFRHead(nn.Module):
    def __init__(self, tab_dim, text_dim, img_dim):
        super().__init__()
        self.nam = NAM(tab_dim)
        self.text_proj = nn.Linear(text_dim, 1)
        self.img_proj = nn.Linear(img_dim, 1)
        self.fusion = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=1))
        self.out = nn.Linear(1, 1)

    def forward(self, tab, txt, img, attn):
        tab_score, tab_contribs = self.nam(tab)
        txt_score = self.text_proj(txt)
        img_score = self.img_proj(img)
        scores = torch.cat([tab_score, txt_score, img_score], dim=1)
        weights = self.fusion(scores)
        fused = torch.sum(scores * weights, dim=1, keepdim=True)
        return self.out(fused), tab_contribs, attn, weights
