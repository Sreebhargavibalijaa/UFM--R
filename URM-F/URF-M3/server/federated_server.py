import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ufm2r.models.text_encoder import InterpretableAttention
from ufm2r.models.reasoning_head import UFM2ReasoningHead

class UFM2Model(nn.Module):
    def __init__(self, tabular_input_dim):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.embed_dim = self.transformer.model.decoder.embed_tokens.embedding_dim
        self.attn = InterpretableAttention(self.embed_dim)
        self.reasoning_head = UFM2ReasoningHead(tabular_input_dim, self.embed_dim)

    def forward(self, tabular_input, input_ids):
        embeddings = self.transformer.model.decoder.embed_tokens(input_ids)
        attn_weights = self.attn(embeddings)
        text_repr = embeddings[:, 0, :]
        return self.reasoning_head(tabular_input, text_repr)
