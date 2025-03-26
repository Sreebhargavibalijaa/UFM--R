import torch
def explain_prediction(model, tokenizer, tab, ids, tab_cols):
    model.eval()
    with torch.no_grad():
        out, contribs = model(tab, ids)
        prob = torch.sigmoid(out).item()
        print(f"\n🔮 Prediction: {prob:.4f}")
        print("\n📊 Feature Contributions:")
        for i, c in enumerate(contribs[0]):
            print(f"  {tab_cols[i]}: {c.item():.4f}")
