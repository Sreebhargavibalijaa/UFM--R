### ✅ File: models/global_model.py

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TABULAR_FEATURES = 10  # ⨀ Set this to your actual number of tabular features

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Identity()

        self.tabular_encoder = nn.Sequential(
            nn.Linear(NUM_TABULAR_FEATURES, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, tabular_data, image_data):
        # Get 7x7 spatial features
        feat_map = self.image_model.conv1(image_data)
        feat_map = self.image_model.bn1(feat_map)
        feat_map = self.image_model.relu(feat_map)
        feat_map = self.image_model.maxpool(feat_map)
        feat_map = self.image_model.layer1(feat_map)
        feat_map = self.image_model.layer2(feat_map)
        feat_map = self.image_model.layer3(feat_map)
        feat_map = self.image_model.layer4(feat_map)

        patch_contrib = feat_map.mean(dim=1)  # shape: [B, 7, 7]
        flat_img_feat = torch.flatten(self.image_model.avgpool(feat_map), 1)

        tab_feat = self.tabular_encoder(tabular_data)
        combined = torch.cat((flat_img_feat, tab_feat), dim=1)
        output = self.classifier(combined)

        return output, tab_feat, patch_contrib



def load_global_model(path="models/global_model.pt"):
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
