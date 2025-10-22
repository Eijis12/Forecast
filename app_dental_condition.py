# ==============================================
# app_dental_condition.py
# ==============================================

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os
from datetime import datetime

# Initialize Flask
app = Flask(__name__)

# 1️⃣ Load dataset and model
DATA_PATH = "dental_conditions_grouped.csv"
LOG_PATH = "diagnosis_logs.csv"

print("🦷 Loading dental condition dataset...")
df = pd.read_csv(DATA_PATH)

# Verify columns
if "Symptoms" not in df.columns or "Diagnosis" not in df.columns:
    raise ValueError("CSV must contain 'Symptoms' and 'Diagnosis' columns.")

symptoms = df["Symptoms"].astype(str).tolist()
diagnoses = df["Diagnosis"].astype(str).tolist()

print("🔹 Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode clinic symptoms once (for fast inference)
embeddings = model.encode(symptoms, convert_to_tensor=True, normalize_embeddings=True)
print(f"✅ Model and dataset ready — {len(symptoms)} records encoded.")

# 2️⃣ Helper function
def recommend_diagnosis(user_input, top_k=3, threshold=0.6):
    user_emb = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(user_emb, embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        score_val = float(score)
        if score_val >= threshold:  # apply confidence filter
            results.append({
                "diagnosis": diagnoses[idx],
                "similarity": score_val,
                "example_symptom": symptoms[idx]
            })
    return results

# 3️⃣ Logging function
def log_query(user_input, results):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "predictions": "; ".join([f"{r['diagnosis']} ({r['similarity']:.2f})" for r in results])
    }
    log_df = pd.DataFrame([log_entry])
    header = not os.path.exists(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", index=False, header=header)

# 4️⃣ API route
@app.route("/predict_diagnosis", methods=["POST"])
def predict_diagnosis():
    data = request.get_json()
    user_symptom = data.get("symptom", "").strip()

    if not user_symptom:
        return jsonify({"error": "No symptom provided"}), 400

    top_matches = recommend_diagnosis(user_symptom)

    # Log every query
    log_query(user_symptom, top_matches)

    # Handle no matches found
    if not top_matches:
        return jsonify({"message": "No close match found. Please describe your symptom differently."}), 200

    return jsonify(top_matches)

# 5️⃣ Health check route (optional)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Dental diagnosis API running."})

# 6️⃣ Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
