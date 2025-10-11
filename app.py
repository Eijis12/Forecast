from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import io
import traceback
import os

app = Flask(__name__)

# ✅ Allow all CORS (for Render frontend)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# =====================================================
# Utility: Generate revenue forecast from Excel
# =====================================================
def generate_forecast():
    excel_path = "DentalRecords_RevenueForecasting.xlsx"
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at {excel_path}")

    df = pd.read_excel(excel_path)

    # Basic validation
    if "Date" not in df.columns or "Revenue" not in df.columns:
        raise ValueError("Excel must contain 'Date' and 'Revenue' columns")

    # Convert and clean date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date")

    # Create time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Prepare data for LightGBM
    X = df[["Year", "Month", "Day", "DayOfWeek"]]
    y = df["Revenue"]

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }

    model = lgb.train(params, train_data, num_boost_round=100)

    # Generate next 12 months forecast
    last_date = df["Date"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Year": [d.year for d in future_dates],
        "Month": [d.month for d in future_dates],
        "Day": [d.day for d in future_dates],
        "DayOfWeek": [d.dayofweek for d in future_dates],
    })

    future_df["Forecasted_Revenue"] = model.predict(future_df[["Year", "Month", "Day", "DayOfWeek"]])

    return future_df


# =====================================================
# Endpoint: Forecast
# =====================================================
@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def forecast_revenue():
    try:
        # Handle empty POST bodies safely
        _ = request.get_json(silent=True) or {}

        forecast_df = generate_forecast()
        forecast_data = forecast_df.to_dict(orient="records")

        return jsonify({
            "status": "success",
            "forecast": forecast_data
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =====================================================
# Endpoint: Forecast History (stub data or can be real)
# =====================================================
@app.route("/api/revenue/history", methods=["GET"])
def revenue_history():
    try:
        # For now, return sample data — replace with actual saved forecasts if needed
        sample_data = [
            {"Date": "2025-09-01", "Forecasted_Revenue": 12450.0},
            {"Date": "2025-10-01", "Forecasted_Revenue": 13280.0},
        ]
        return jsonify({
            "status": "success",
            "history": sample_data
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =====================================================
# Health Check
# =====================================================
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Revenue Forecasting API is running!",
        "endpoints": [
            "/api/revenue/forecast (POST)",
            "/api/revenue/history (GET)"
        ]
    })


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
