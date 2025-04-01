# client_model.py
import torch.nn as nn
from nam import ImagePatchNAM

class UFMClientModel(nn.Module):
    def __init__(self, num_patches=100):
        super().__init__()
        self.image_encoder = ImagePatchNAM(num_patches=num_patches)
        self.classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary output
        )

    def forward(self, img):
        score, patch_contribs = self.image_encoder(img)
        out = self.classifier(score)
        return out, patch_contribs
