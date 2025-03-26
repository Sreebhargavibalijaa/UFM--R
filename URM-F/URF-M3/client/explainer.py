def explain_prediction(model, tokenizer, tab, ids, tabular_cols):
    model.eval()
    with torch.no_grad():
        output, contribs = model(tab, ids)
        prob = torch.sigmoid(output).item()
        print(f"\n🔮 Prediction: {prob:.4f}")
        print("\n📊 Tabular Feature Contributions:")
        for i, val in enumerate(contribs[0]):
            print(f"  {tabular_cols[i]}: {val.item():.4f}")
