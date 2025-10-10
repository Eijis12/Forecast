from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# ✅ 1. DIAGNOSIS PREDICTION SETUP
# =========================================================
FILE_PATH = os.path.join(os.path.dirname(__file__), "dental_health_forecasting.xlsx")
df = pd.read_excel(FILE_PATH)

# Basic cleaning
df = df.dropna(subset=["Diagnosis"])
df["Symptoms"] = df["Symptoms"].fillna("")
df["Treatment"] = df["Treatment"].fillna("")
df["input_text"] = df["Symptoms"] + " " + df["Treatment"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    df["input_text"], df["Diagnosis"], test_size=0.2, random_state=42
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

unique_treatments = sorted(df["Treatment"].dropna().unique().tolist())

@app.route("/")
def home():
    return jsonify({"message": "Dental Forecast ML API is running 🚀"})

@app.route("/api/treatments", methods=["GET"])
def treatments():
    return jsonify(unique_treatments)

@app.route("/api/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    treatment_input = data.get("treatment", "").strip()
    symptom_input = data.get("symptom", "").strip()
    if not symptom_input and not treatment_input:
        return jsonify({"error": "Please provide at least one symptom or treatment."}), 400

    input_text = f"{symptom_input} {treatment_input}".strip()
    predicted_diagnosis = model.predict([input_text])[0]
    confidence = round(float(max(model.predict_proba([input_text])[0])), 2)
    return jsonify({
        "predicted_diagnosis": predicted_diagnosis,
        "confidence": confidence
    })

# =========================================================
# ✅ 2. REVENUE FORECASTING ENDPOINT
# =========================================================
REVENUE_FILE = os.path.join(os.path.dirname(__file__), "DentalRecords_RevenueForecasting.xlsx")

def forecast_next_month(file_path=REVENUE_FILE, steps=30):
    df = pd.read_excel(file_path)
    amount_col = [c for c in df.columns if 'amount' in c.lower()][0]

    if 'DATE' not in df.columns:
        if all(col in df.columns for col in ['YEAR', 'MONTH']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        else:
            raise ValueError("No DATE or YEAR/MONTH columns found in dataset.")

    monthly = df.groupby(pd.Grouper(key='DATE', freq='M'))[amount_col].sum().sort_index()
    rows = []
    for m, total in monthly.items():
        days = pd.date_range(start=m.to_period('M').to_timestamp(), end=m + pd.offsets.MonthEnd(0), freq='D')
        share = total / len(days)
        for d in days:
            rows.append((d, share))
    daily = pd.DataFrame(rows, columns=['DATE', 'AMOUNT']).set_index('DATE')

    np.random.seed(0)
    daily['AMOUNT'] += np.random.normal(scale=0.03 * daily['AMOUNT'].mean(), size=len(daily))
    daily['AMOUNT'] = np.maximum(daily['AMOUNT'], 0)

    model = SARIMAX(daily['AMOUNT'], order=(1,1,1), seasonal_order=(1,1,1,7))
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
    total_forecast = mean_forecast.sum()

    return {
        "next_month_total": round(total_forecast, 2),
        "daily_forecast": mean_forecast.round(2).to_dict(),
        "confidence_intervals": conf_int.round(2).to_dict()
    }

@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    try:
        result = forecast_next_month()
        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
