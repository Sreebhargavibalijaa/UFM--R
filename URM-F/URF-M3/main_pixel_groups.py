import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------
# NAM Block
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
# Text Attention
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
# Image Patch Encoder (20 groups)
# -----------------------
import math

class ImagePatchNAM(nn.Module):
    def __init__(self, num_patches=20):
        super().__init__()
        self.num_patches = num_patches
        self.rows, self.cols = self._factorize_patches(num_patches)
        self.patch_nam = NAM(num_patches)

    def _factorize_patches(self, n):
        # Return (rows, cols) such that rows * cols = n
        for i in range(int(math.sqrt(n)), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n  # fallback

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        patches = []
        ph, pw = H // self.rows, W // self.cols

        for i in range(self.rows):
            for j in range(self.cols):
                patch = x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                pooled = patch.reshape(B, -1).mean(dim=1, keepdim=True)
                patches.append(pooled)

        patch_feats = torch.cat(patches, dim=1)  # [B, num_patches]
        score, contribs = self.patch_nam(patch_feats)
        return score, contribs

# -----------------------
# NAFR Head
# -----------------------
class NAFRHead(nn.Module):
    def __init__(self, tab_dim, text_dim, img_dim):
        super().__init__()
        self.nam = NAM(tab_dim)
        self.text_proj = nn.Linear(text_dim, 1)
        self.fusion = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=1))
        self.out = nn.Linear(1, 1)

    def forward(self, tab, txt, img_score, attn_weights):
        tab_score, tab_contribs = self.nam(tab)
        text_score = self.text_proj(txt)
        scores = torch.cat([tab_score, text_score, img_score], dim=1)
        weights = self.fusion(scores)
        fused = torch.sum(scores * weights, dim=1, keepdim=True)
        return self.out(fused), tab_contribs, attn_weights, weights

# -----------------------
# Full Model
# -----------------------
class UFM3Model(nn.Module):
    def __init__(self, tab_dim, num_patches):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        dim = self.lm.model.decoder.embed_tokens.embedding_dim
        self.attn = InterpretableAttention(dim)
        self.img_encoder = ImagePatchNAM(num_patches=100)
        self.head = NAFRHead(tab_dim, dim, img_dim=1)

    def forward(self, tab, ids, img):
        text_embed = self.lm.model.decoder.embed_tokens(ids)
        attn = self.attn(text_embed)
        text_feat = text_embed[:, 0, :]
        img_score, img_contribs = self.img_encoder(img)
        out, tab_contribs, attn_weights, fusion = self.head(tab, text_feat, img_score, attn)
        return out, tab_contribs, attn_weights, fusion, img_contribs

# -----------------------
# Plotting Utils
# -----------------------
def plot_tabular_contributions(contribs, labels):
    values = [c.item() for c in contribs[0]]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Tabular Feature Contributions")
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

def plot_patch_overlay_on_image(contribs, rows, cols, original_img_tensor, filename="patch_overlay.png"):
    import cv2

    # Step 1: Prepare contribution heatmap
    contrib_array = contribs[0].detach().cpu().numpy().reshape(rows, cols)
    contrib_resized = cv2.resize(contrib_array, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Normalize for visualization
    norm_map = (contrib_resized - contrib_resized.min()) / (contrib_resized.max() - contrib_resized.min() + 1e-8)

    # Step 2: Get original image in H×W×3 format
    img_np = original_img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = img_np / img_np.max()  # normalize

    # Step 3: Convert to RGB if grayscale
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)

    # Step 4: Overlay with heatmap
    heatmap = plt.cm.jet(norm_map)[:, :, :3]  # remove alpha channel
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Step 5: Plot + save
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title("🩻 Chest X-ray Patch Contributions (NAM)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved overlay: {filename}")


# -----------------------
# Run Demo
# -----------------------
def run_demo():
    print("🧠 Running UFM³-R with Patch NAM for Image...")

    tab = torch.tensor([[65.0, 160.0, 110.0]])
    text = "Patient reports severe chest pain and shortness of breath."

    model = UFM3Model(tab_dim=3, num_patches=500)
    tokenizer = model.tokenizer
    input_ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]

    img_path = "assets/demo.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError("🖼️ Please add demo.jpg to assets/")

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        out, tab_contribs, attn, fusion, img_contribs = model(tab, input_ids, img_tensor)
        prob = torch.sigmoid(out).item()

    print(f"\n✅ Prediction: {prob:.4f} → {'Positive' if prob > 0.5 else 'Negative'}")

    print("\n📊 Tabular Feature Contributions:")
    labels = ["Age", "BP", "HR"]
    for i, c in enumerate(tab_contribs[0]):
        print(f"  {labels[i]}: {c.item():.4f}")
    # plot_tabular_contributions(tab_contribs, labels)

    print("\n📝 Top Tokens by Attention:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attn[0][:len(tokens)].detach().cpu().numpy()
    top_tokens = [(tokens[i], scores[i]) for i in range(len(tokens)) if tokens[i] != "<pad>"]
    top_tokens = sorted(top_tokens, key=lambda x: x[1], reverse=True)[:10]
    for t, s in top_tokens:
        print(f"  {t}: {s:.4f}")
    plot_token_attention([t for t, _ in top_tokens], [s for _, s in top_tokens])

    print("\n Image Patch Contributions:")
    for i, val in enumerate(img_contribs[0]):
        print(f"  Patch {i+1:02d}: {val.item():.4f}")
    plot_patch_overlay_on_image(img_contribs, model.img_encoder.rows, model.img_encoder.cols, img_tensor)


    print("\n⚖️ Fusion Weights:")
    print(f"  Tabular: {fusion[0,0]:.4f}")
    print(f"  Text:    {fusion[0,1]:.4f}")
    print(f"  Image:   {fusion[0,2]:.4f}")

if __name__ == "__main__":
    run_demo()
