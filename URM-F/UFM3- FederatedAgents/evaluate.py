# evaluate.py

import os, random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

from models.global_model import HybridModel

NUM_TABULAR_FEATURES = 10
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChestXrayDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        tab_tensor = torch.rand(NUM_TABULAR_FEATURES)
        return tab_tensor, img_tensor, label

def evaluate_global_model(client_dirs):
    model = HybridModel().to(device)
    model.load_state_dict(torch.load("models/global_model.pt", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    all_data = []
    for folder in client_dirs:
        for label, subfolder in enumerate(["normal", "pneumonia"]):
            sub_path = os.path.join(folder, subfolder)
            for file in os.listdir(sub_path):
                all_data.append((os.path.join(sub_path, file), label))

    random.shuffle(all_data)
    dataset = ChestXrayDataset(all_data, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    y_true, y_pred = [], []
    with torch.no_grad():
        for tab, img, labels in loader:
            tab, img = tab.to(device), img.to(device)
            outputs, _, _ = model(tab, img)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
