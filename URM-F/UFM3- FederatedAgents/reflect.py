def reflect_on_disagreement(agent_outputs,confidences, report):
    """
    Analyzes disagreements among agent predictions and returns a reflection string.
    """
    from collections import Counter

    predictions = [d["decision"] for d in agent_outputs]
    confidences = [d["prob"] for d in agent_outputs]
    counter = Counter(predictions)
    majority = counter.most_common(1)[0]

    if len(counter) == 1:
        return "✅ All agents agree. High reliability of the diagnosis."
    elif all(p < 0.55 for p in confidences):
        return "🔴 All agents have low confidence. Suggest additional data or imaging."
    else:
        return f"⚠️ Disagreement detected: {dict(counter)}. Majority voted for '{majority[0]}'. Consider reviewing with human expert."
    
