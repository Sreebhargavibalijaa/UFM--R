# ✅ Final `train_federated.py` — Manual Client Folder Paths via Function

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.global_model import HybridModel
import torch.optim as optim

# --- Config ---
NUM_TABULAR_FEATURES = 10
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loader ---
def load_client_data(folder_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(folder_path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Simulated Tabular ---
def simulate_tabular(batch_size):
    return torch.rand(batch_size, NUM_TABULAR_FEATURES)

# --- Train One Client ---
def train_one_client(model, dataloader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        tabular = simulate_tabular(images.size(0)).to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(tabular, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model.state_dict()

# --- Weight Aggregation ---
def average_weights(w_list):
    return {k: sum(w[k] for w in w_list) / len(w_list) for k in w_list[0]}

# --- Main Federated Training Call ---
def train_federated_model_from_paths(client_paths, save_path="models/global_model.pt"):
    weights = []
    for path in client_paths:
        dataloader = load_client_data(path)
        model = HybridModel().to(device)
        client_weights = train_one_client(model, dataloader, device)
        weights.append(client_weights)

    global_model = HybridModel().to(device)
    global_model.load_state_dict(average_weights(weights))
    os.makedirs("models", exist_ok=True)
    torch.save(global_model.state_dict(), save_path)
    print(f"✅ Global model saved at: {save_path}")

# import os, random
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from models.global_model import HybridModel
# model = HybridModel()

# NUM_TABULAR_FEATURES = 10
# NUM_CLIENTS = 3
# EPOCHS = 2
# BATCH_SIZE = 16
# LR = 0.001

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pneumonia_path = "all_pneumonia"
# normal_path = "all_normal"

# full_dataset = []
# for label, folder in enumerate([normal_path, pneumonia_path]):
#     for file in os.listdir(folder):
#         full_dataset.append((os.path.join(folder, file), label))
# random.shuffle(full_dataset)
# client_data = [full_dataset[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

# class ChestXrayDataset(Dataset):
#     def __init__(self, data, transform):
#         self.data = data
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         path, label = self.data[idx]
#         img = Image.open(path).convert("RGB")
#         img_tensor = self.transform(img)

#         tab_tensor = torch.rand(NUM_TABULAR_FEATURES)  # Simulate tabular data
#         return tab_tensor, img_tensor, label

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# def train(model, dataloader, optimizer, criterion):
#     model.train()
#     for tab, img, labels in dataloader:
#         tab, img, labels = tab.to(device), img.to(device), labels.to(device)
#         optimizer.zero_grad()
#         out, _, _ = model(tab, img)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()

# def get_weights(model):
#     return {k: v.cpu().clone() for k, v in model.state_dict().items()}

# def set_weights(model, weights):
#     model.load_state_dict(weights)

# def average_weights(w_list):
#     return {k: sum(w[k] for w in w_list) / len(w_list) for k in w_list[0]}

# global_model = HybridModel().to(device)
# criterion = nn.CrossEntropyLoss()

# for round in range(EPOCHS):
#     local_weights = []
#     for cid in range(NUM_CLIENTS):
#         dataset = ChestXrayDataset(client_data[cid], transform)
#         loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#         local_model = HybridModel().to(device)
#         set_weights(local_model, get_weights(global_model))

#         optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
#         train(local_model, loader, optimizer, criterion)
#         local_weights.append(get_weights(local_model))

#     avg_weights = average_weights(local_weights)
#     set_weights(global_model, avg_weights)

# os.makedirs("models", exist_ok=True)
# torch.save(global_model.state_dict(), "models/global_model.pt")
# print("✅ Global model saved.")

