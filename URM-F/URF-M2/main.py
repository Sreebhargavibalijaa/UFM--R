# main_ufm2r.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -----------------------
# NAM for Tabular Data
# -----------------------

class FeatureNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

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
# Text Attention Block
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
        return self.softmax(scores)[:, 0, :]  # CLS attention

# -----------------------
# UFM²-R Model
# -----------------------

class UFM2RModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        embed_dim = self.lm.model.decoder.embed_tokens.embedding_dim
        self.attn = InterpretableAttention(embed_dim)
        self.nam = NAM(tab_dim)
        self.text_proj = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(2, 1)

    def forward(self, tab, ids):
        embed = self.lm.model.decoder.embed_tokens(ids)
        attn = self.attn(embed)
        txt = embed[:, 0, :]
        tab_score, contribs = self.nam(tab)
        text_score = self.text_proj(txt)
        fused = self.fc(torch.cat([tab_score, text_score], dim=1))
        return fused, contribs, attn

# -----------------------
# Data Prep Function
# -----------------------

def compute_tabular(example):
    text = example["text"]
    words = text.split()
    return {
        "length": len(text),
        "avg_word_len": sum(len(w) for w in words) / len(words) if words else 0,
        "exclamations": text.count("!")
    }

# -----------------------
# Run Demo
# -----------------------

def run_demo():
    print("🧠 Running UFM²-R demo...")

    # Load data (IMDB-like sample)
    dataset = load_dataset("imdb")["train"].map(compute_tabular)
    sample = dataset[0]
    tab = torch.tensor([[sample["length"], sample["avg_word_len"], sample["exclamations"]]], dtype=torch.float32)
    text = sample["text"]

    # Tokenize text
    model = UFM2RModel(tab_dim=3)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]

    # Run model
    with torch.no_grad():
        out, contribs, attn = model(tab, ids)
        prob = torch.sigmoid(out).item()

    print(f"\n Prediction: {prob:.4f} → {'Positive' if prob >= 0.5 else 'Negative'}")

    print("\n📊 Feature Contributions:")
    feature_names = ["Length", "Avg Word Length", "Exclamations"]
    for i, val in enumerate(contribs[0]):
        print(f"  {feature_names[i]}: {val.item():.4f}")


    print("\n Top Contributing Tokens:")
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    scores = attn[0][:len(tokens)].detach().cpu().numpy()
    top_idx = scores.argsort()[-5:][::-1]
    for i in top_idx:
        print(f"  {tokens[i]}: {scores[i]:.4f}")

if __name__ == "__main__":
    run_demo()
