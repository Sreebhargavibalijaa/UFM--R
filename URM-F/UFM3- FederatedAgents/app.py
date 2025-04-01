# ✅ Final `app.py` — Unified Federated Multimodal Sustainable Diagnosis System

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import os
import time
import platform
from datetime import datetime

from agent import FederatedUFMSystem
from utils import plot_patch_overlay_on_image
from train_federated import train_federated_model_from_paths
from evaluate import evaluate_global_model
from reflect import reflect_on_disagreement

# --- App Config ---
st.set_page_config(page_title="UFM Federated Multimodal Sustainable Detector", layout="wide")
st.title("Unified Federated Multimodal Sustainable Diagnosis System")
st.markdown("Upload chest X-rays, enter patient data, and get explainable pneumonia diagnosis across multiple AI agents.")

# --- Sidebar Inputs ---
st.sidebar.header("\U0001F3E5 Client Dataset Paths")
client1_path = st.sidebar.text_input("Client 1 Folder Path")
client2_path = st.sidebar.text_input("Client 2 Folder Path")
client3_path = st.sidebar.text_input("Client 3 Folder Path")
init_train = st.sidebar.button("\U0001F680 Train Federated Model from Clients")
eval_model = st.sidebar.button("\U0001F4CA Evaluate Global Model")

# --- Federated Training Trigger ---
if init_train:
    if all([client1_path, client2_path, client3_path]):
        st.info("\U0001F9E0 Training Federated Model...")

        start_time = time.time()
        train_federated_model_from_paths([client1_path, client2_path, client3_path])
        duration = time.time() - start_time

        device_used = "GPU" if torch.cuda.is_available() else "CPU"
        emissions_kg = round(duration * 0.0004, 4)
        trees_needed = round(emissions_kg / 21.77, 3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("assets", exist_ok=True)
        report_text = f"""\U0001F4CA Carbon Report — {timestamp}

\U0001F5A5️ Device Used: {device_used}
⏱️ Runtime: {duration:.2f} sec
\U0001F30D CO₂ Emissions: {emissions_kg} kg
\U0001F333 Trees to Offset: {trees_needed}
"""
        report_path = f"assets/train_run_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        st.success("✅ Federated Model Trained!")
        st.markdown(f"**\U0001F4BB Device:** `{device_used}`")
        st.markdown(f"**⏱ Runtime:** `{duration:.2f}` seconds")
        st.markdown(f"**\U0001F30D CO₂ Emissions:** `{emissions_kg} kg`")
        st.markdown(f"**\U0001F333 Trees to Offset:** `{trees_needed}`")
        st.download_button("⬇️ Download Carbon Report", data=report_text, file_name=f"carbon_report_{timestamp}.txt", mime="text/plain")
    else:
        st.error("❌ Please provide all 3 client dataset paths.")

# --- Global Model Evaluation ---
if eval_model:
    if all([client1_path, client2_path, client3_path]):
        st.info("\U0001F4CA Evaluating global model...")
        eval_results = evaluate_global_model([client1_path, client2_path, client3_path])
        st.success("✅ Evaluation Complete")
        st.markdown(f"**Accuracy:** `{eval_results['accuracy']:.2f}`")
        st.markdown(f"**Precision:** `{eval_results['precision']:.2f}`")
        st.markdown(f"**Recall:** `{eval_results['recall']:.2f}`")
        st.markdown(f"**F1 Score:** `{eval_results['f1']:.2f}`")
    else:
        st.error("❌ Please provide all 3 client dataset paths.")

# --- Patient Input ---
with st.sidebar:
    st.header("\U0001F9CD Patient Info")
    patient_id = st.text_input("Patient ID", value="patient_001")
    age = st.number_input("Age", 0, 120, 65)
    bp = st.number_input("Blood Pressure", 80, 200, 160)
    hr = st.number_input("Heart Rate", 40, 180, 110)
    report = st.text_area("\U0001F4DD Clinical Report", "Patient reports chest pain and shortness of breath.")
    uploaded_image = st.file_uploader("\U0001FA7B Upload Chest X-ray (.jpg/.png)", type=["png", "jpg"])
    run_analysis = st.button("\U0001F50D Run Federated Diagnosis")

# --- Inference Flow ---
if uploaded_image and run_analysis:
    img = Image.open(uploaded_image).convert("RGB")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    tab_tensor = torch.tensor([[age, bp, hr]], dtype=torch.float32)

    fed = FederatedUFMSystem(num_agents=3)
    with st.spinner("Running inference across agents..."):
        results = fed.run_all(tab_tensor, report, img_tensor, patient_id)

    # --- Display Results ---
    st.subheader("\U0001F916 Agent Predictions")
    col1, col2 = st.columns([1, 2])
    decisions = []
    confidences = []
    with col1:
        for name, decision, prob in results:
            st.markdown(f"**{name}**\n- Prediction: `{decision}`\n- Probability: `{prob:.4f}`")
            decisions.append(decision)
            confidences.append(prob)
    with col2:
        st.image(uploaded_image, caption="Uploaded Chest X-ray", use_container_width=True)

    # --- Final Clinical Inference ---
    st.subheader("\U0001F9E0 Final Clinical Inference")
    majority_label = max(set(decisions), key=decisions.count)
    majority_diagnosis = "Pneumonia" if majority_label.lower() != "normal" else "Normal"
    st.success(f"✅ Majority Diagnosis: **{majority_diagnosis}**")

#     if len(set(decisions)) > 1:
#         st.warning("⚠️ Agents Disagree. Manual review advised.")
#         agent_outputs = [{"name": name, "decision": decision, "prob": prob} for name, decision, prob in results]
# # reflection = reflect_on_disagreement(agent_outputs, report)
#         reflection = reflect_on_disagreement(decisions, confidences, report)
#         st.markdown(f"**🧠 Reflective Agent Insight:** {reflection}")
#     if all(conf < 0.55 for conf in confidences):
#         st.error("🔴 All agents have low confidence (<0.55). Diagnosis is uncertain.")
        # st.markdown("📌 Suggest more patient data or further tests.")

    # --- Agent-1 Insights ---
    st.subheader("\U0001F9EA AI Diagnosis (Agent-1)")
    prob = confidences[0]
    st.success(f"🤖 Agent-1 Diagnosis: {'Pneumonia' if prob > 0.5 else 'Normal'}")
    st.write(f"**Confidence Score:** `{prob:.4f}`")

    st.subheader("\U0001FA7B Patch-Level Contribution (Agent-1)")
    img_contribs = fed.agents[0].memory.last(patient_id)["img_contribs"]
    overlay_path = plot_patch_overlay_on_image(img_contribs, 10, 10, img_tensor)
    st.image(overlay_path, caption="Patch Overlay Heatmap")

    st.subheader("\U0001F4DD Text Attention (Agent-1)")
    attn = fed.agents[0].memory.last(patient_id)["attn"]
    tokenizer = fed.agents[0].tokenizer
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(report, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"][0]
    )
    scores = attn[0].detach().cpu().numpy().flatten()
    top_tokens = sorted([(tokens[i], scores[i]) for i in range(min(len(tokens), len(scores))) if tokens[i] != "<pad>"],
                        key=lambda x: x[1], reverse=True)[:10]
    for token, score in top_tokens:
        st.write(f"`{token}`: {score:.4f}")

    st.subheader("\U0001F4CA Tabular Feature Contributions (Agent-1)")
    tab_contribs = fed.agents[0].memory.last(patient_id)["tab_contribs"]
    feature_names = ["Age", "BP", "HR"]
    for i, val in enumerate(tab_contribs[0]):
        scalar = val.flatten()[0].item() if val.numel() > 1 else val.item()
        st.write(f"- **{feature_names[i]}**: `{scalar:.4f}`")

    # --- Feedback Loop ---
    st.subheader("\U0001F4AC Feedback")
    feedback = st.text_input("Provide feedback or clarification to agents:")
    if feedback:
        st.success("✅ Feedback received (simulated memory update).")

    # --- Memory Viewer ---
    st.subheader("\U0001F9E0 Agent Memory Viewer")
    memory_tabs = st.tabs([agent.name for agent in fed.agents])
    for i, agent in enumerate(fed.agents):
        with memory_tabs[i]:
            memory = agent.memory.summary(patient_id)
            if not memory:
                st.info("No memory recorded yet.")
            else:
                for idx, interaction in enumerate(memory):
                    st.markdown(f"""
                    **Interaction {idx+1}**
                    - Prediction: `{interaction['prediction']}`
                    - Probability: `{interaction['probability']:.4f}`
                    - Notes: _{interaction['notes']}_
                    """)
                if st.button(f"🧹 Clear Memory ({agent.name})", key=f"clear-{i}"):
                    agent.memory.memory.pop(patient_id, None)
                    st.success(f"{agent.name}'s memory has been cleared.")

# --- Footer ---
st.markdown("---")
st.markdown("🔬 Built for Chest X-ray Pneumonia Detection | Federated Training | Carbon-Aware Intelligence")