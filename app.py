from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
import io
import traceback
import os
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"

# ==========================================================
# Forecast generation
# ==========================================================
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    # ✅ Handle REVENUE or AMOUNT
    if "REVENUE" in df.columns and "AMOUNT" not in df.columns:
        df.rename(columns={"REVENUE": "AMOUNT"}, inplace=True)

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {required}. Found: {df.columns.tolist()}")

    for col in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df["Revenue"] = df["AMOUNT"]
    df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date")

    if df.empty:
        raise ValueError("No valid data rows found. Please check your Excel file values.")

    # === Exponential Smoothing ===
    try:
        exp_model = ExponentialSmoothing(df["Revenue"], trend="add", seasonal=None)
        exp_fit = exp_model.fit()
        exp_forecast = exp_fit.forecast(30)
    except Exception:
        exp_forecast = np.repeat(df["Revenue"].mean(), 30)

    # === Prophet ===
    prophet_df = df.rename(columns={"Date": "ds", "Revenue": "y"})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future_prophet = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future_prophet)
    prophet_forecast = prophet_forecast.tail(30)["yhat"].values

    # === LightGBM ===
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    X = df[["Year", "Month", "Day", "DayOfWeek"]]
    y = df["Revenue"]

    lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=120, learning_rate=0.1)
    lgb_model.fit(X, y)

    # === Forecast next 30 days ===
    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]

    future_X = pd.DataFrame({
        "Year": [d.year for d in future_dates],
        "Month": [d.month for d in future_dates],
        "Day": [d.day for d in future_dates],
        "DayOfWeek": [d.dayofweek for d in future_dates],
    })
    lgb_forecast = lgb_model.predict(future_X)

    # === Hybrid combination ===
    combined_forecast = (0.3 * exp_forecast) + (0.3 * prophet_forecast) + (0.4 * lgb_forecast)

    # === Sundays closed ===
    day_of_week = [d.dayofweek for d in future_dates]
    combined_forecast = [0 if dow == 6 else val for dow, val in zip(day_of_week, combined_forecast)]

    # === Combine historical + forecast ===
    hist_df = df[["Date", "Revenue"]].copy()
    hist_df["Type"] = "historical"

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Revenue": combined_forecast,
        "Type": "forecast"
    })

    combined_df = pd.concat([hist_df, future_df], ignore_index=True)

    # === Metrics (MAE only) ===
    try:
        mae = mean_absolute_error(df["Revenue"].iloc[-30:], exp_fit.fittedvalues[-30:])
    except Exception:
        mae = float("nan")

    total_forecast = np.sum(combined_forecast)

    forecast_result = {
        "chart_data": [
            {"date": str(row["Date"].date()), "revenue": round(row["Revenue"], 2), "type": row["Type"]}
            for _, row in combined_df.iterrows()
        ],
        "daily_forecast": {
            str(d.date()): round(r, 2)
            for d, r in zip(future_dates, combined_forecast)
        },
        "total_forecast": round(total_forecast, 2),
        "mae": round(mae, 2) if not np.isnan(mae) else None,
    }

    return forecast_result

# ==========================================================
# API Routes
# ==========================================================
@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API active"})

@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def forecast_revenue():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        result = generate_forecast()
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/revenue/history", methods=["GET", "OPTIONS"])
def get_history():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        return jsonify({"status": "success", "data": []})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        df = pd.DataFrame(
            list(result["daily_forecast"].items()), columns=["Date", "Forecasted_Revenue"]
        )
        df.loc[len(df)] = ["Total (₱)", result["total_forecast"]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast_30_Days")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================================
# Run App (Render-compatible)
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


