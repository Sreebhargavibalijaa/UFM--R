import os, random
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Config
NUM_CLIENTS = 3
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
pneumonia_path = "all_pneumonia"
normal_path = "all_normal"
assert os.path.exists(pneumonia_path) and os.path.exists(normal_path), "Folders not found!"

# Load all image paths and labels
full_dataset = []
for label, folder in enumerate([normal_path, pneumonia_path]):
    for file in os.listdir(folder):
        full_dataset.append((os.path.join(folder, file), label))
random.shuffle(full_dataset)

# Split into 3 clients
client_data = [full_dataset[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

# Dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# CNN model (ResNet18)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Federated training functions
def train(model, dataloader, optimizer, criterion):
    model.train()
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

def get_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_weights(model, weights):
    model.load_state_dict(weights)

def average_weights(weight_list):
    return {k: sum(w[k] for w in weight_list) / len(weight_list) for k in weight_list[0]}

# Train
global_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

for round in range(EPOCHS):
    local_weights = []
    for cid in range(NUM_CLIENTS):
        data = client_data[cid]
        dataset = ChestXrayDataset(data, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        local_model = SimpleCNN().to(device)
        set_weights(local_model, get_weights(global_model))

        optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
        train(local_model, loader, optimizer, criterion)
        local_weights.append(get_weights(local_model))
        print(f"✅ Client {cid+1} finished Round {round+1}")

    avg_weights = average_weights(local_weights)
    set_weights(global_model, avg_weights)
    print(f"🔁 Global model updated after Round {round+1}\n")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(global_model.state_dict(), "models/global_model.pt")
print("✅ Global model saved to models/global_model.pt")
