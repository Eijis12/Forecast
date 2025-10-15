from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import io
import traceback
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from prophet import Prophet

# =====================================================
# Flask App Configuration
# =====================================================
app = Flask(__name__)

FRONTEND_ORIGIN = "https://campbelldentalsystem.site"  # Change if needed
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}}, supports_credentials=True)


@app.after_request
def add_cors_headers(response):
    """Ensure all responses have the correct CORS headers"""
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# =====================================================
# Helper: Recursive Forecast Function
# =====================================================
def generate_recursive_forecast(df, forecast_days=30):
    # Ensure correct format
    df = df.rename(columns={"Date": "ds", "Revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    # Train Prophet model (trend + seasonality)
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    prophet_model.fit(df)

    # Predict future trend
    future_dates = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future_dates)
    df["prophet_trend"] = prophet_forecast["yhat"][:len(df)]

    # Prepare LightGBM features
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["trend"] = df["prophet_trend"]
    df = df.dropna()

    X = df[["day_of_week", "month", "lag_1", "lag_7", "trend"]]
    y = df["y"]

    # Train LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)

    # Recursive Forecasting
    last_date = df["ds"].max()
    future_data = df.copy()

    for i in range(forecast_days):
        next_date = last_date + timedelta(days=i + 1)
        next_day_of_week = next_date.weekday()
        next_month = next_date.month

        next_trend = prophet_model.predict(pd.DataFrame({"ds": [next_date]}))["yhat"].values[0]

        last_row = future_data.iloc[-1]
        lag_1 = last_row["y"]
        lag_7 = future_data.iloc[-7]["y"] if len(future_data) >= 7 else lag_1

        features = np.array([[next_day_of_week, next_month, lag_1, lag_7, next_trend]])
        next_pred = model.predict(features)[0]

        future_data = pd.concat([
            future_data,
            pd.DataFrame([{
                "ds": next_date,
                "y": next_pred,
                "day_of_week": next_day_of_week,
                "month": next_month,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "trend": next_trend
            }])
        ], ignore_index=True)

    forecast_df = future_data.tail(forecast_days)[["ds", "y"]].rename(columns={"ds": "Date", "y": "Forecasted_Revenue"})

    # Validation MAE (last 20%)
    split_idx = int(len(df) * 0.8)
    y_true = df["y"].iloc[split_idx:]
    y_pred = model.predict(X.iloc[split_idx:])
    mae = mean_absolute_error(y_true, y_pred)

    return forecast_df, mae


# =====================================================
# API Route: Forecast
# =====================================================
@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
@cross_origin(origin=FRONTEND_ORIGIN, supports_credentials=True)
def forecast_revenue():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        excel_path = "DentalRecords_RevenueForecasting.xlsx"

        if not os.path.exists(excel_path):
            return jsonify({"status": "error", "message": "Excel file not found."}), 404

        df = pd.read_excel(excel_path)
        if "Date" not in df.columns or "Revenue" not in df.columns:
            return jsonify({"status": "error", "message": "Missing columns in Excel file."}), 400

        # Generate recursive forecast
        forecast_df, mae = generate_recursive_forecast(df, forecast_days=30)

        # Save forecast for download
        output_path = "forecast_results.xlsx"
        forecast_df.to_excel(output_path, index=False)

        total_forecast = forecast_df["Forecasted_Revenue"].sum()

        return jsonify({
            "status": "success",
            "mae": round(mae, 2),
            "total_forecast": round(total_forecast, 2),
            "forecast": forecast_df.to_dict(orient="records")
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =====================================================
# API Route: Download Forecast
# =====================================================
@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    file_path = "forecast_results.xlsx"
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "No forecast file found."}), 404

    return send_file(file_path, as_attachment=True)


# =====================================================
# Root Endpoint
# =====================================================
@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API Running Successfully"}), 200


# =====================================================
# Run App
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
