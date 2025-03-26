import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Tabular: NAM
# -----------------------
class FeatureNet(nn.Module):
    def __init__(self, hidden=32):
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

# -----------------------
# Text: Attention
# -----------------------
class InterpretableAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q, k = self.q(x), self.k(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        return self.softmax(scores)[:, 0, :]

# -----------------------
# Image Encoder
# -----------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.pool(feat).view(x.size(0), -1)
        return self.project(pooled)

# -----------------------
# NAFR Fusion Head
# -----------------------
class NAFRHead(nn.Module):
    def __init__(self, tab_dim, text_dim, img_dim):
        super().__init__()
        self.nam = NAM(tab_dim)
        self.text_proj = nn.Linear(text_dim, 1)
        self.img_proj = nn.Linear(img_dim, 1)
        self.fusion = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=1))
        self.out = nn.Linear(1, 1)

    def forward(self, tab, txt, img, attn_weights):
        tab_score, contribs = self.nam(tab)
        text_score = self.text_proj(txt)
        img_score = self.img_proj(img)
        scores = torch.cat([tab_score, text_score, img_score], dim=1)
        weights = self.fusion(scores)
        fused = torch.sum(scores * weights, dim=1, keepdim=True)
        return self.out(fused), contribs, attn_weights, weights

# -----------------------
# Full UFM³-R Model
# -----------------------
class UFM3Model(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        dim = self.lm.model.decoder.embed_tokens.embedding_dim
        self.attn = InterpretableAttention(dim)
        self.img_encoder = ImageEncoder()
        self.head = NAFRHead(tab_dim, dim, 128)

    def forward(self, tab, ids, img):
        text_embed = self.lm.model.decoder.embed_tokens(ids)
        attn = self.attn(text_embed)
        text_feat = text_embed[:, 0, :]
        img_feat = self.img_encoder(img)
        return self.head(tab, text_feat, img_feat, attn)

# -----------------------
# Grad-CAM Wrapper
# -----------------------
class ImageCAMWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)

# -----------------------
# Plotting Utils
# -----------------------
def plot_tabular_contributions(contribs, labels):
    values = [c.item() for c in contribs[0]]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("📊 Tabular Feature Contributions")
    plt.ylabel("Contribution")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("tabular_contributions.png")
    plt.show()

def plot_token_attention(tokens, scores):
    plt.figure(figsize=(10, 3))
    bars = plt.bar(tokens, scores, color='salmon')
    plt.title("📝 Token Attention Scores")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Attention")
    plt.tight_layout()
    plt.savefig("token_attention.png")
    plt.show()

# -----------------------
# Demo Runner
# -----------------------
def run_demo():
    print("🧠 Running UFM³-R Demo (Text + Tabular + Image)...")

    tab = torch.tensor([[65.0, 160.0, 110.0]])
    text = "Patient reports severe chest pain and shortness of breath."

    model = UFM3Model(tab_dim=3)
    tokenizer = model.tokenizer
    input_ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]

    img_path = "assets/demo.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError("🖼️ Add demo.jpg inside assets/")

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output, contribs, attn, fusion = model(tab, input_ids, img_tensor)
        prob = torch.sigmoid(output).item()
    print(f"\n✅ Prediction: {prob:.4f} → {'Positive' if prob > 0.5 else 'Negative'}")

    # 📊 Tabular
    tab_labels = ["Age", "Blood Pressure", "Heart Rate"]
    print("\n📊 Tabular Feature Contributions:")
    for i, c in enumerate(contribs[0]):
        print(f"  {tab_labels[i]}: {c.item():.4f}")
    plot_tabular_contributions(contribs, tab_labels)

    # 📝 Text
    print("\n📝 Top Tokens by Attention:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attn[0][:len(tokens)].detach().cpu().numpy()
    top_tokens = [(tokens[i], scores[i]) for i in range(len(tokens)) if tokens[i] != "<pad>"]
    top_tokens = sorted(top_tokens, key=lambda x: x[1], reverse=True)[:10]
    for t, s in top_tokens:
        print(f"  {t}: {s:.4f}")
    plot_token_attention([t for t, _ in top_tokens], [s for _, s in top_tokens])

    # ⚖️ Fusion Weights
    print("\n⚖️ Modality Fusion Weights:")
    print(f"  Tabular: {fusion[0,0]:.4f}")
    print(f"  Text:    {fusion[0,1]:.4f}")
    print(f"  Image:   {fusion[0,2]:.4f}")

    # 🧠 Grad-CAM for Image
    print("\n🖼️ Generating Grad-CAM...")
    cam_model = ImageCAMWrapper(model.img_encoder)
    cam_model.train()
    cam_extractor = GradCAM(model=cam_model, target_layer=cam_model.encoder.encoder[7])
    img_tensor.requires_grad = True
    with torch.enable_grad():
        scores = cam_model(img_tensor)
        class_idx = scores.argmax().item()
        cam_map = cam_extractor(class_idx, scores)

    cam_array = cam_map[0].detach().cpu().numpy()
    if cam_array.ndim == 3:
        cam_array = cam_array.squeeze(0)

    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    cam_resized = T.Resize((img_np.shape[0], img_np.shape[1]))(torch.tensor(cam_array)).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(cam_resized, cmap='jet', alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("Grad-CAM Intensity")
    plt.title("🧠 Image Contribution (Grad-CAM)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("cam_colormap_output.png")
    plt.show()
    print("📷 Grad-CAM heatmap saved → cam_colormap_output.png")

if __name__ == "__main__":
    run_demo()
