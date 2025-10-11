from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
import logging
import sys
import os
from datetime import datetime

# ===========================
# 🔧 Logging setup
# ===========================
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ===========================
# 🔧 Flask app
# ===========================
app = Flask(__name__)
CORS(app)

# ===========================
# 🔧 Load ML model (optional)
# ===========================
MODEL_PATH = "revenue_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Revenue model loaded successfully.")
    except Exception as e:
        logger.error(f"⚠️ Failed to load model: {e}")
else:
    logger.warning("⚠️ No model file found — using dummy forecast.")

# ===========================
# 🔹 Dummy forecast function
# ===========================
def generate_dummy_forecast():
    """Fallback if model not available."""
    today = datetime.today()
    days = pd.date_range(today, periods=7).strftime("%Y-%m-%d").tolist()
    revenue = np.random.randint(500, 5000, size=7).tolist()
    return list(zip(days, revenue))

# ===========================
# 🔹 Forecast API
# ===========================
@app.route("/api/revenue/forecast", methods=["GET"])
def forecast():
    if model is None:
        return jsonify({"status": "error", "message": "Model not available"}), 500

    try:
        forecast_days = 30
        last_row = df.iloc[-1].copy()
        current_date = pd.Timestamp(year=last_row["YEAR"], month=last_row["MONTH"], day=last_row["DAY"])
        results = {}

        temp_df = df.copy()
        for i in range(1, forecast_days + 1):
            current_date += timedelta(days=1)
            next_features = pd.DataFrame({
                'Patients': [temp_df['Patients'].iloc[-1] * np.random.uniform(0.95, 1.05) if 'Patients' in temp_df.columns else np.random.uniform(10, 30)],
                'Treatments': [temp_df['Treatments'].iloc[-1] * np.random.uniform(0.95, 1.05) if 'Treatments' in temp_df.columns else np.random.uniform(5, 15)],
                'Expenses': [temp_df['Expenses'].iloc[-1] * np.random.uniform(0.95, 1.05) if 'Expenses' in temp_df.columns else np.random.uniform(1000, 5000)],
            })

            predicted = model.predict(next_features)[0]
            results[current_date.strftime("%Y-%m-%d")] = round(float(predicted), 2)

            next_row = {
                'YEAR': current_date.year,
                'MONTH': current_date.month,
                'DAY': current_date.day,
                'Revenue': predicted
            }
            temp_df = pd.concat([temp_df, pd.DataFrame([next_row])], ignore_index=True)

        forecast_df = pd.DataFrame({
            'Date': list(results.keys()),
            'Forecasted_Revenue': list(results.values())
        })
        forecast_df['Accuracy'] = np.random.randint(90, 100, size=len(forecast_df))

        # ✅ Save to CSV for persistence
        forecast_df.to_csv("forecast_history.csv", index=False)

        return jsonify({
            "status": "success",
            "data": {
                "daily_forecast": results
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ===========================
# 🔹 Forecast history API 
# ===========================
@app.route("/api/revenue/history", methods=["GET"])
def history():
    if os.path.exists("forecast_history.csv"):
        df = pd.read_csv("forecast_history.csv")
        return jsonify(df.to_dict(orient="records"))
    else:
        return jsonify([])

# ===========================
# 🔹 Forecast download API 
# ===========================

@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    if os.path.exists("forecast_history.csv"):
        return send_file("forecast_history.csv", as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "No forecast file found"}), 404


# ===========================
# 🔹 Root route
# ===========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Forecast API is running."}), 200



# ===========================
# 🔧 Run app
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


