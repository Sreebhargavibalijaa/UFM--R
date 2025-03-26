# 🤖 UFM²-R: Unified Federated Multi-Modal Foundation Model with Interpretable Reasoning

**FedMM-X** is a federated learning framework built for **multi-modal data** — combining **image (MNIST)**, **text (IMDB)**, and **tabular** features — with a focus on privacy, trust, and interpretability.

## 🔍 Key Features

- ✅ **Privacy-Preserving Learning**: Data never leaves the client.
- 🧠 **Multi-Modal Model**: Unified neural network processes image, text, and tabular inputs.
- 🎯 **Trust-Weighted Aggregation**: Clients contribute based on reliability scores.
- 📈 **FedAvg vs. FedMM-X**: Compare aggregation strategies and accuracy across clients.
- 💬 **LIME Interpretability**: Generate HTML explanations showing which tabular features influenced predictions.

## 🌐 Project Workflow

1. Load and align MNIST + IMDB data into multi-modal format.
2. Split data across clients with assigned trust levels.
3. Train using both **FedAvg** and **FedMM-X** methods.
4. Evaluate client accuracy and explainability using **LIME**.
5. Visualize results with comparison plots and explanation dashboards.

## 📊 Live Visualization

A **dynamic global graph** tracks each client’s accuracy and interpretability contribution during training. This helps identify which clients drive global performance and where explanations align or diverge.

> ⚡ Coming soon: Streamlit dashboard for real-time federated interpretability!

## 📁 Outputs

- `checkpoints/` – Saved FedAvg and FedMM-X models
- `lime_explanation_client_*.html` – Feature attribution visualizations per client
- `accuracy_comparison.png` – Bar chart comparing FedAvg and FedMM-X

## 🛠️ Requirements

```bash
pip install torch torchvision flwr datasets lime matplotlib
