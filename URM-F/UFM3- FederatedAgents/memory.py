# memory.py — Final Version with Updated Summary Format

class AgentMemory:
    def __init__(self):
        self.memory = {}

    def store(self, patient_id, record):
        if patient_id not in self.memory:
            self.memory[patient_id] = []
        self.memory[patient_id].append(record)

    def last(self, patient_id):
        if patient_id in self.memory and self.memory[patient_id]:
            return self.memory[patient_id][-1]
        return None

    def summary(self, patient_id):
        if patient_id not in self.memory:
            return []
        return [
            {
                "prediction": "Chest X-ray" if r["prob"] > 0.5 else "no pneumonia",
                "probability": r["prob"],
                "tab_contribs": r["tab_contribs"],
                "img_contribs": r["img_contribs"],
                "notes": r["text"]
            }
            for r in self.memory[patient_id]
        ]
