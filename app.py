from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load dataset
FILE_PATH = os.path.join(os.path.dirname(__file__), "dental_health_forecasting.xlsx")
df = pd.read_excel(FILE_PATH)

# ✅ Basic cleaning
df = df.dropna(subset=["Diagnosis"])
df["Symptoms"] = df["Symptoms"].fillna("")
df["Treatment"] = df["Treatment"].fillna("")
df["input_text"] = df["Symptoms"] + " " + df["Treatment"]

# ✅ Split data (for internal validation — not necessary for prediction)
X_train, X_test, y_train, y_test = train_test_split(
    df["input_text"], df["Diagnosis"], test_size=0.2, random_state=42
)

# ✅ Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# ✅ Train model
model.fit(X_train, y_train)

# ✅ For dropdowns
unique_treatments = sorted(df["Treatment"].dropna().unique().tolist())

@app.route("/")
def home():
    return jsonify({"message": "Dental Forecast ML API is running 🚀"})

@app.route("/treatments", methods=["GET"])
def treatments():
    return jsonify(unique_treatments)

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    treatment_input = data.get("treatment", "").strip()
    symptom_input = data.get("symptom", "").strip()

    if not symptom_input and not treatment_input:
        return jsonify({"error": "Please provide at least one symptom or treatment."}), 400

    # Combine both text fields for prediction
    input_text = f"{symptom_input} {treatment_input}".strip()

    # ✅ Predict diagnosis
    predicted_diagnosis = model.predict([input_text])[0]
    confidence = round(float(max(model.predict_proba([input_text])[0])), 2)

    return jsonify({
        "predicted_diagnosis": predicted_diagnosis,
        "confidence": confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
