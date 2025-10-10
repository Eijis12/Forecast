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
import traceback

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

    # --- Handle date/time ---
    if 'DATE' not in df.columns:
        if all(col in df.columns for col in ['YEAR', 'MONTH']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        else:
            raise ValueError("No DATE or YEAR/MONTH columns found in dataset.")

    # --- Group to daily totals ---
    daily = df.groupby(pd.Grouper(key='DATE', freq='D'))[amount_col].sum().fillna(0)
    daily = daily.asfreq('D').fillna(method='ffill')

    # --- FAST FORECAST MODEL (simple trend + rolling mean) ---
    rolling_mean = daily.rolling(window=7, min_periods=1).mean()
    trend = np.linspace(1, 1.05, len(rolling_mean))  # simple upward trend
    adjusted = rolling_mean * trend

    last_value = adjusted.iloc[-1]
    forecast_values = np.linspace(last_value, last_value * 1.05, steps)  # small linear increase

    forecast_dates = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')

    forecast_df = pd.Series(forecast_values, index=forecast_dates)
    total_forecast = forecast_df.sum()

    conf_lower = forecast_df * 0.9
    conf_upper = forecast_df * 1.1

    # --- ✅ Convert datetime index to strings for JSON ---
    forecast_dict = {d.strftime("%Y-%m-%d"): float(v) for d, v in forecast_df.items()}
    conf_lower_dict = {d.strftime("%Y-%m-%d"): float(v) for d, v in conf_lower.items()}
    conf_upper_dict = {d.strftime("%Y-%m-%d"): float(v) for d, v in conf_upper.items()}

    return {
        "next_month_total": round(total_forecast, 2),
        "daily_forecast": forecast_dict,
        "confidence_intervals": {
            "lower": conf_lower_dict,
            "upper": conf_upper_dict
        }
    }


@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    try:
        print("⚙️ Starting revenue forecast...")
        result = forecast_next_month()
        print("✅ Forecast complete!")
        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": result
        })
    except Exception as e:
        print("❌ Forecast failed:")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# ✅ 3. MAIN APP ENTRY POINT
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
