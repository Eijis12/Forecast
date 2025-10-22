# ==============================================
# app_dental_condition.py
# ==============================================

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Initialize Flask
app = Flask(__name__)

# 1️⃣ Load dataset and model
df = pd.read_csv("dental_conditions_grouped.csv")
symptoms = df["Symptoms"].astype(str).tolist()
diagnoses = df["Diagnosis"].astype(str).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(symptoms, convert_to_tensor=True, normalize_embeddings=True)

print(f"✅ Model and dataset loaded — {len(symptoms)} records")

# 2️⃣ Define helper function
def recommend_diagnosis(user_input, top_k=3):
    user_emb = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(user_emb, embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "diagnosis": diagnoses[idx],
            "similarity": float(score),
            "example_symptom": symptoms[idx]
        })
    return results

# 3️⃣ API route
@app.route("/predict_diagnosis", methods=["POST"])
def predict_diagnosis():
    data = request.get_json()
    user_symptom = data.get("symptom", "")
    if not user_symptom:
        return jsonify({"error": "No symptom provided"}), 400

    top_matches = recommend_diagnosis(user_symptom)
    return jsonify(top_matches)

# 4️⃣ Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
