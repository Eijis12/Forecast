from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import traceback
import os

app = Flask(__name__)

# ✅ Allow only your site (this fixes the multiple header issue)
CORS(app, resources={r"/api/*": {"origins": "https://campbelldentalsystem.site"}})

# --- Helper: parse month safely ---
def parse_month(value):
    try:
        m = pd.to_datetime(str(value), errors="coerce", format="%B")
        if pd.isna(m):
            m = pd.to_datetime(str(value), errors="coerce", format="%b")
        if pd.isna(m):
            val = str(value).strip().upper()
            month_map = {
                "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
                "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
                "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12
            }
            return month_map.get(val, np.nan)
        return m.month
    except Exception:
        return np.nan


# --- Forecast Route ---
@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def forecast_revenue():
    try:
        print("🔹 Forecast API triggered")

        excel_path = "DentalRecords_RevenueForecasting.xlsx"
        if not os.path.exists(excel_path):
            return jsonify({"error": f"File not found: {excel_path}"}), 404

        # Load Excel data
        df = pd.read_excel(excel_path)

        # --- Clean & Prepare ---
        df = df.rename(columns=lambda x: x.strip())
        if "Month" not in df.columns or "Revenue" not in df.columns:
            return jsonify({"error": "Excel must contain 'Month' and 'Revenue' columns"}), 400

        df["Month_Num"] = df["Month"].apply(parse_month)
        df = df.dropna(subset=["Month_Num", "Revenue"])

        df["Month_Num"] = df["Month_Num"].astype(int)
        df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
        df = df.dropna(subset=["Revenue"])

        # --- Train model ---
        X = df[["Month_Num"]]
        y = df["Revenue"]

        model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1)
        model.fit(X, y)

        # --- Forecast next 6 months ---
        last_month = df["Month_Num"].max()
        future_months = [(last_month + i - 1) % 12 + 1 for i in range(1, 7)]
        forecast_df = pd.DataFrame({"Month_Num": future_months})
        forecast_df["Predicted_Revenue"] = model.predict(forecast_df[["Month_Num"]])

        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        forecast_df["Month"] = forecast_df["Month_Num"].apply(lambda x: month_names[x - 1])

        result = forecast_df[["Month", "Predicted_Revenue"]].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "forecast": result,
            "note": "Forecast generated successfully"
        })

    except Exception as e:
        print("❌ Error:", traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500


# --- Root route for Render test ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Revenue Forecast API is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
