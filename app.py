from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import io
import traceback
from datetime import datetime
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------------------------------------
# Helper function for forecasting
# -------------------------------------------------------
def forecast_revenue(model_type="hybrid"):
    df = pd.read_excel("DentalRecords_RevenueForecasting.xlsx")
    df.rename(columns={"Date": "ds", "Revenue": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    # Sundays = 0 (clinic closed)
    df.loc[df["ds"].dt.dayofweek == 6, "y"] = 0

    # Split train/test (last 14 days for validation)
    train = df[:-14]
    test = df[-14:]
    future_periods = 30

    # ---------- Exponential Smoothing ----------
    exp_model = ExponentialSmoothing(train["y"], trend="add", seasonal="add", seasonal_periods=7).fit()
    exp_pred = exp_model.forecast(14)

    # ---------- Prophet ----------
    prophet = Prophet()
    prophet.fit(train)
    future_test = prophet.make_future_dataframe(periods=14, include_history=False)
    prophet_pred = prophet.predict(future_test)["yhat"]

    # ---------- LightGBM ----------
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    X_train = df.loc[df.index < len(train), ["dayofweek", "month"]]
    y_train = train["y"]
    X_test = df.loc[df.index >= len(train), ["dayofweek", "month"]]
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    lightgbm_pred = model.predict(X_test)

    # ---------- Combine Hybrid ----------
    hybrid_pred = (exp_pred.values + prophet_pred.values + lightgbm_pred) / 3

    # ---------- MAE Calculation ----------
    mae = mean_absolute_error(test["y"], hybrid_pred)

    # ---------- Forecast Next 30 Days ----------
    exp_forecast = exp_model.forecast(future_periods)
    future_full = prophet.make_future_dataframe(periods=future_periods)
    prophet_forecast = prophet.predict(future_full).tail(future_periods)

    future_dates = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=future_periods)
    future_features = pd.DataFrame({
        "dayofweek": future_dates.dayofweek,
        "month": future_dates.month
    })
    lightgbm_forecast = model.predict(future_features)

    hybrid_future = (exp_forecast.values + prophet_forecast["yhat"].values + lightgbm_forecast) / 3

    results = pd.DataFrame({
        "Date": future_dates,
        "Exp_Smoothing": exp_forecast.values,
        "Prophet": prophet_forecast["yhat"].values,
        "LightGBM": lightgbm_forecast,
        "Hybrid": hybrid_future
    })

    # ---------- Save History ----------
    os.makedirs("history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_excel(f"history/forecast_{timestamp}.xlsx", index=False)

    return results, mae


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.route("/api/revenue/forecast", methods=["GET"])
def get_forecast():
    try:
        results, mae = forecast_revenue()
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            results.to_excel(writer, index=False)
        output.seek(0)

        preview = results.head(5).to_dict(orient="records")
        return jsonify({
            "message": "Forecast generated successfully",
            "mae": mae,
            "preview": preview
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/revenue/history", methods=["GET"])
def get_history():
    try:
        if not os.path.exists("history"):
            return jsonify({"files": []})
        files = sorted(os.listdir("history"), reverse=True)
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/revenue/download/<filename>", methods=["GET"])
def download_file(filename):
    try:
        return send_file(os.path.join("history", filename), as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Revenue Forecast API running successfully."})


# -------------------------------------------------------
# Production entry point
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
