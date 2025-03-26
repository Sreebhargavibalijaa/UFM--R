import torch
from torch.utils.data import Dataset

class UFM2Dataset(Dataset):
    def __init__(self, dataset, tokenizer, tabular_cols):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tabular_cols = tabular_cols

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        tabular = torch.tensor([row[col] for col in self.tabular_cols], dtype=torch.float32)
        tokens = self.tokenizer(row["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        input_ids = tokens["input_ids"].squeeze(0)
        label = torch.tensor(row["label"], dtype=torch.float32).unsqueeze(0)
        return tabular, input_ids, label
