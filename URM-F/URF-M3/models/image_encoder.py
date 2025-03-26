import torch.nn as nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.feat = nn.Sequential(*list(base.children())[:-1])
        self.project = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.project(self.feat(x).view(x.size(0), -1))
