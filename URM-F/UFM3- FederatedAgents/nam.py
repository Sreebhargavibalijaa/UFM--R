# nam.py
import torch
import torch.nn as nn
import math

class FeatureNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x)

class NAM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_nets = nn.ModuleList([FeatureNet() for _ in range(input_dim)])
    def forward(self, x):
        contribs = [net(x[:, i].unsqueeze(1)) for i, net in enumerate(self.feature_nets)]
        contribs = torch.cat(contribs, dim=1)
        return contribs.sum(dim=1, keepdim=True), contribs

class ImagePatchNAM(nn.Module):
    def __init__(self, num_patches=100):
        super().__init__()
        self.rows, self.cols = self._factorize(num_patches)
        self.patch_nam = NAM(num_patches)

    def _factorize(self, n):
        for i in range(int(math.sqrt(n)), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        ph, pw = H // self.rows, W // self.cols
        patches = []

        for i in range(self.rows):
            for j in range(self.cols):
                patch = x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                pooled = patch.view(B, -1).mean(dim=1, keepdim=True)
                patches.append(pooled)

        feats = torch.cat(patches, dim=1)  # [B, num_patches]
        score, contribs = self.patch_nam(feats)
        return score, contribs
