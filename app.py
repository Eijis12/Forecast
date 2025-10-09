from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from difflib import SequenceMatcher, get_close_matches
import os

app = Flask(__name__)

# ✅ Allow CORS from anywhere (so Hostinger frontend can connect)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load dataset from the same directory (works on Render)
FILE_PATH = os.path.join(os.path.dirname(__file__), "dental_health_forecasting.xlsx")
df = pd.read_excel(FILE_PATH)

# ✅ Clean treatments
def clean_treatments(treatments):
    unique = []
    for t in treatments:
        t = str(t).strip().lower()
        if not t or t == "nan":
            continue
        close = get_close_matches(t, unique, n=1, cutoff=0.85)
        if not close:
            unique.append(t)
    return sorted([t.title() for t in unique])

cleaned_treatments = clean_treatments(df["Treatment"].dropna().unique())

# 🔮 Forecast route
@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    year = data.get("year")
    month = data.get("month")
    treatment_input = data.get("treatment", "").strip().lower()
    symptom_input = data.get("symptom", "").strip().lower()

    if not treatment_input and not symptom_input:
        return jsonify({"error": "Missing treatment or symptom"}), 400

    # 🧠 Compute similarity based on both inputs (symptom more important)
    df["Treatment_Similarity"] = df["Treatment"].fillna("").apply(
        lambda t: SequenceMatcher(None, treatment_input, t.lower()).ratio()
    )
    df["Symptom_Similarity"] = df["Symptoms"].fillna("").apply(
        lambda s: SequenceMatcher(None, symptom_input, s.lower()).ratio()
    )

    # Weighted more toward symptom similarity
    df["Overall_Similarity"] = (df["Treatment_Similarity"] * 0.4) + (df["Symptom_Similarity"] * 0.6)

    # Pick matches above a lower threshold to allow flexible matches
    similar_rows = df[df["Overall_Similarity"] > 0.2]

    if similar_rows.empty:
        # Try again using only symptom similarity
        symptom_only = df[df["Symptom_Similarity"] > 0.25]
        if not symptom_only.empty:
            top_match = symptom_only.sort_values("Symptom_Similarity", ascending=False).iloc[0]
            diagnosis = top_match["Diagnosis"]
            confidence = round(float(top_match["Symptom_Similarity"]), 2)
            return jsonify({
                "predicted_diagnosis": diagnosis,
                "confidence": confidence
            })
        # No match at all
        return jsonify({
            "predicted_diagnosis": "General dental check-up",
            "confidence": 0.0
        })

    # ✅ Get the best match from combined similarity
    top_match = similar_rows.sort_values("Overall_Similarity", ascending=False).iloc[0]
    diagnosis = top_match["Diagnosis"]
    confidence = round(float(top_match["Overall_Similarity"]), 2)

    return jsonify({
        "predicted_diagnosis": diagnosis,
        "confidence": confidence
    })


@app.route("/")
def home():
    return jsonify({"message": "Dental Forecast API is running 🚀"})


# ✅ Return cleaned treatment list
@app.route("/treatments", methods=["GET"])
def get_treatments():
    return jsonify(cleaned_treatments)

if __name__ == "__main__":
    # Render sets PORT automatically — use that if available
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


