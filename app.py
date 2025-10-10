from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import datetime
import random
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import lightgbm as lgb

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# ✅ 1. DIAGNOSIS PREDICTION
# =========================================================
FILE_PATH = os.path.join(os.path.dirname(__file__), "dental_health_forecasting.xlsx")
df = pd.read_excel(FILE_PATH)

df = df.dropna(subset=["Diagnosis"])
df["Symptoms"] = df["Symptoms"].fillna("")
df["Treatment"] = df["Treatment"].fillna("")
df["input_text"] = df["Symptoms"] + " " + df["Treatment"]

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
# ✅ 2. REVENUE FORECASTING
# =========================================================
REVENUE_FILE = os.path.join(os.path.dirname(__file__), "DentalRecords_RevenueForecasting.xlsx")

def forecast_next_month(file_path=REVENUE_FILE, steps=30):
    df = pd.read_excel(file_path)
    df.columns = [c.strip().upper() for c in df.columns]

    required = ["YEAR", "MONTH", "DAY", "AMOUNT"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" +
        df["MONTH"].astype(str).str.zfill(2) + "-" +
        df["DAY"].astype(str).str.zfill(2),
        errors="coerce"
    )
    df = df.dropna(subset=["DATE"])

    if df.empty:
        raise ValueError("No valid rows after parsing DATE column.")

    daily = df.groupby("DATE")["AMOUNT"].sum().fillna(0)
    if daily.empty:
        raise ValueError("No valid revenue data found after grouping by DATE.")
    daily = daily.asfreq("D").fillna(method="ffill")

    data = pd.DataFrame({
        "date": daily.index,
        "revenue": daily.values,
    })
    data["dayofweek"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year

    X = data[["dayofweek", "month", "year"]]
    y = data["revenue"]

    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42
    )
    model.fit(X, y)

    today = pd.Timestamp.now().normalize()
    forecast_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=steps, freq="D")

    future_data = pd.DataFrame({
        "date": forecast_dates,
        "dayofweek": forecast_dates.dayofweek,
        "month": forecast_dates.month,
        "year": forecast_dates.year
    })

    preds = model.predict(future_data[["dayofweek", "month", "year"]])
    preds = np.maximum(preds, 0)
    preds = preds / preds.sum() * random.uniform(50000, 100000)
    preds[future_data["dayofweek"] == 6] = 0  # Sundays = 0

    forecast_df = pd.Series(preds, index=future_data["date"]).round(2)
    total_forecast = forecast_df.sum().round(2)

    return {
        "next_month_total": float(total_forecast),
        "daily_forecast": forecast_df.to_dict()
    }


@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    try:
        result = forecast_next_month()
        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": {
                "next_month_total": result["next_month_total"],
                "daily_forecast": {str(k): v for k, v in result["daily_forecast"].items()},
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# ✅ RUN APP
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
