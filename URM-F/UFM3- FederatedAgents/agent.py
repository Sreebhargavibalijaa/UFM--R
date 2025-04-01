# agent.py — Now includes reflection, planner, and dynamic modality importance

import torch
from transformers import AutoTokenizer
from model import UFM3Model
from memory import AgentMemory
import random

class UFM3Agent:
    def __init__(self, name="Agent-A", threshold=0.5):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        self.model = UFM3Model(tab_dim=3, num_patches=100, tokenizer=self.tokenizer)
        self.threshold = threshold
        self.memory = AgentMemory()

    def perceive(self, patient_id, tab, text, img_tensor, modality_hint="auto"):
        ids = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]
        with torch.no_grad():
            out, tab_contribs, attn, fusion, img_contribs = self.model(tab, ids, img_tensor)
            prob = torch.sigmoid(out).item()

        # Dynamic explanation prioritization
        dominant_modality = self.get_dominant_modality(tab, text, modality_hint)

        result = {
            "text": text,
            "tab": tab,
            "img": img_tensor,
            "prob": prob,
            "tab_contribs": tab_contribs,
            "attn": attn,
            "fusion": fusion,
            "img_contribs": img_contribs,
            "dominant_modality": dominant_modality
        }
        self.memory.store(patient_id, result)
        return result

    def decide(self, prob):
        return "Pneumonia" if prob > self.threshold else "Normal"

    def explain(self, record):
        return {
            "tabular": record["tab_contribs"],
            "attention": record["attn"],
            "fusion": record["fusion"],
            "img_contribs": record["img_contribs"]
        }

    def get_dominant_modality(self, tab, text, hint):
        if hint != "auto":
            return hint
        if tab[0][0].item() > 70:
            return "tabular"
        elif len(text.split()) > 25:
            return "text"
        else:
            return "image"

class FederatedUFMSystem:
    def __init__(self, num_agents=3):
        self.agents = [UFM3Agent(name=f"Agent-{i+1}") for i in range(num_agents)]

    def run_all(self, tab, text, image_tensor, patient_id):
        decisions = []
        for agent in self.agents:
            modality_hint = self.plan_modality(tab, text)
            result = agent.perceive(patient_id, tab, text, image_tensor, modality_hint=modality_hint)
            decision = agent.decide(result["prob"])
            decisions.append((agent.name, decision, result["prob"]))
        return decisions

    def plan_modality(self, tab, text):
        # Planner logic: decide dominant modality hint
        if tab[0][0].item() > 70:
            return "tabular"
        elif len(text.split()) > 25:
            return "text"
        return "image"

    def reflect_on_disagreement(self, results):
        decisions = [res[1] for res in results]
        if len(set(decisions)) > 1:
            return "⚠️ Disagreement Detected. LLM reflection recommended."
        return "✅ All agents agree."

# Extension point for future LLM reflection integration
# def query_reflection_agent(agents_results):
#     # Use OpenAI or Claude to summarize disagreement and reasoning
#     pass