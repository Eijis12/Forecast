from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import datetime
import io
import lightgbm as lgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HISTORY_FILE = "forecast_history.xlsx"

# ==========================================================
# Utility: ensure history file exists
# ==========================================================
def initialize_history():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["Date", "Actual", "Predicted", "Model", "MAE"])
        df.to_excel(HISTORY_FILE, index=False)

initialize_history()

# ==========================================================
# Utility: recursive hybrid forecasting function
# ==========================================================
def hybrid_recursive_forecast(df, forecast_days=30):
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds")

    # Prophet Model
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet.fit(df[["ds", "y"]])

    # LightGBM Features
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["lag1"] = df["y"].shift(1)
    df["lag7"] = df["y"].shift(7)
    df = df.dropna()

    features = ["dayofweek", "month", "lag1", "lag7"]
    X_train, y_train = df[features], df["y"]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model_lgb.fit(X_train, y_train)

    # Recursive Forecasting
    future_dates = []
    last_known_date = df["ds"].max()
    last_known_values = df.copy()

    for i in range(forecast_days):
        next_date = last_known_date + datetime.timedelta(days=1)
        dayofweek = next_date.weekday()
        month = next_date.month

        # skip Sundays (revenue = 0)
        if dayofweek == 6:
            next_y = 0
        else:
            lag1 = last_known_values.iloc[-1]["y"]
            lag7 = last_known_values.iloc[-7]["y"] if len(last_known_values) >= 7 else lag1

            # Prophet prediction for next date
            prophet_pred = prophet.predict(pd.DataFrame({"ds": [next_date]}))["yhat"].values[0]

            # LightGBM prediction using lags
            next_x = np.array([[dayofweek, month, lag1, lag7]])
            lgb_pred = model_lgb.predict(next_x)[0]

            # hybrid blend
            next_y = 0.5 * prophet_pred + 0.5 * lgb_pred

        future_dates.append({"ds": next_date, "y": next_y})
        new_row = pd.DataFrame({"ds": [next_date], "y": [next_y]})
        last_known_values = pd.concat([last_known_values, new_row], ignore_index=True)
        last_known_date = next_date

    forecast_df = pd.DataFrame(future_dates)
    return forecast_df


# ==========================================================
# Forecast API
# ==========================================================
@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_revenue():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_excel(file)
        df.columns = [col.strip().upper() for col in df.columns]

        if "DATE" not in df.columns or "REVENUE" not in df.columns:
            return jsonify({"error": "Missing DATE or REVENUE column"}), 400

        df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)

        # Split into train/validation for MAE check
        train_size = int(len(df) * 0.9)
        train_df = df.iloc[:train_size]
        valid_df = df.iloc[train_size:]

        forecast_df = hybrid_recursive_forecast(train_df, forecast_days=len(valid_df))

        mae = mean_absolute_error(valid_df["y"].values, forecast_df["y"].values[: len(valid_df)])

        # Combine actual + forecast for display
        forecast_df["Actual"] = list(valid_df["y"].values[: len(forecast_df)]) + [np.nan] * max(
            0, len(forecast_df) - len(valid_df)
        )
        forecast_df["Model"] = "Hybrid (Prophet + LightGBM)"
        forecast_df["MAE"] = mae

        # Save to Excel history
        initialize_history()
        existing = pd.read_excel(HISTORY_FILE)
        combined = pd.concat(
            [existing, forecast_df[["ds", "Actual", "y", "Model", "MAE"]].rename(columns={"ds": "Date", "y": "Predicted"})],
            ignore_index=True,
        )
        combined.to_excel(HISTORY_FILE, index=False)

        return jsonify(
            {
                "message": "Forecast completed successfully",
                "mae": round(mae, 2),
                "forecast": forecast_df.to_dict(orient="records"),
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ==========================================================
# Download API
# ==========================================================
@app.route("/api/revenue/download", methods=["GET"])
def download_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"error": "No forecast history available"}), 404

    return send_file(HISTORY_FILE, as_attachment=True)


# ==========================================================
# Root
# ==========================================================
@app.route("/")
def home():
    return "Dental Revenue Forecasting API - Hybrid Recursive Model is running!"


# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
