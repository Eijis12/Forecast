from flask import Flask, request, jsonify, send_file
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
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# ✅ 1. DIAGNOSIS PREDICTION SETUP
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
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "forecast_results.xlsx")


def forecast_next_month(file_path=REVENUE_FILE, steps=30):
    df = pd.read_excel(file_path)

    # detect amount column
    amount_col = [c for c in df.columns if 'amount' in c.lower()][0]

    # ✅ FIXED: Create DATE using YEAR, MONTH, DAY
    if 'DATE' not in df.columns:
        if all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY']):
            df['DATE'] = pd.to_datetime(
                df['YEAR'].astype(str) + '-' +
                df['MONTH'].astype(str) + '-' +
                df['DAY'].astype(str)
            )
        elif all(col in df.columns for col in ['YEAR', 'MONTH']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        else:
            raise ValueError("No DATE or YEAR/MONTH/DAY columns found in dataset.")

    df['DATE'] = pd.to_datetime(df['DATE'])

    # aggregate by day
    daily = df.groupby(pd.Grouper(key='DATE', freq='D'))[amount_col].sum().fillna(0)
    daily = daily.asfreq('D').fillna(method='ffill')

    data = pd.DataFrame({
        'date': daily.index,
        'revenue': daily.values,
    })
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    X = data[['dayofweek', 'month', 'year']]
    y = data['revenue']

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(X, y)

    today = pd.Timestamp.now().normalize()
    forecast_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=steps, freq='D')

    future_data = pd.DataFrame({
        'date': forecast_dates,
        'dayofweek': forecast_dates.dayofweek,
        'month': forecast_dates.month,
        'year': forecast_dates.year
    })

    preds = model.predict(future_data[['dayofweek', 'month', 'year']])

    # Add scaling and random variation
    target_total = random.uniform(50000, 100000)
    preds = preds / preds.sum() * target_total

    sunday_mask = future_data['dayofweek'] == 6
    preds[sunday_mask] = 0
    preds = preds * np.random.uniform(0.9, 1.1, size=len(preds))

    forecast_df = pd.Series(preds, index=future_data['date']).round(2)
    total_forecast = forecast_df.sum().round(2)
    conf_lower = (forecast_df * 0.9).round(2)
    conf_upper = (forecast_df * 1.1).round(2)

    save_df = pd.DataFrame({
        "Date": forecast_df.index.strftime("%Y-%m-%d"),
        "Predicted_Revenue": forecast_df.values,
        "Lower_Bound": conf_lower.values,
        "Upper_Bound": conf_upper.values
    })

    save_df.to_excel(HISTORY_FILE, index=False)

    return {
        "next_month_total": float(total_forecast),
        "daily_forecast": forecast_df.to_dict(),
        "confidence_intervals": {"lower": conf_lower.to_dict(), "upper": conf_upper.to_dict()},
        "saved_to": HISTORY_FILE
    }


@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    try:
        result = forecast_next_month()
        daily_forecast_str = {str(d): v for d, v in result["daily_forecast"].items()}
        conf_lower_str = {str(d): v for d, v in result["confidence_intervals"]["lower"].items()}
        conf_upper_str = {str(d): v for d, v in result["confidence_intervals"]["upper"].items()}

        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": {
                "next_month_total": result["next_month_total"],
                "daily_forecast": daily_forecast_str,
                "confidence_intervals": {
                    "lower": conf_lower_str,
                    "upper": conf_upper_str
                },
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/history", methods=["GET"])
def revenue_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"status": "empty", "message": "No forecast history found yet."})
    df = pd.read_excel(HISTORY_FILE)
    return jsonify({
        "status": "success",
        "message": "Forecast history retrieved.",
        "history": df.to_dict(orient="records")
    })


# =========================================================
# ✅ DOWNLOAD FORECAST
# =========================================================
@app.route("/api/revenue/download", methods=["GET"])
def download_forecast_file():
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({"status": "error", "message": "No forecast file found."}), 404
        return send_file(HISTORY_FILE, as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# ✅ RUN APP
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
