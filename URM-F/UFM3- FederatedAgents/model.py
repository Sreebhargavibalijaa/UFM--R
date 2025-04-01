# model.py — Final Dummy UFM3Model (Corrected Signature)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UFM3Model(nn.Module):
    def __init__(self, tab_dim, num_patches, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tab_encoder = nn.Linear(tab_dim, 8)
        self.text_encoder = nn.Embedding(30522, 8)  # Simulated vocab size
        self.img_encoder = nn.Conv2d(3, 1, kernel_size=3, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(8 + 8 + num_patches, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tab, text_ids, img):
        tab_feat = self.tab_encoder(tab)  # (B, 8)
        text_feat = self.text_encoder(text_ids).mean(dim=1)  # (B, 8)
        img_feat = self.img_encoder(img)  # (B, 1, H, W)
        patches = img_feat.view(img_feat.size(0), -1)[:, :100]  # (B, 100)
        fusion = torch.cat([tab_feat, text_feat, patches], dim=1)
        out = self.classifier(fusion)
        tab_contribs = tab_feat.unsqueeze(0)
        attn = F.softmax(text_feat.unsqueeze(0), dim=-1)
        img_contribs = patches.view(1, -1)
        return out, tab_contribs, attn, fusion, img_contribs