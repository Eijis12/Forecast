from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import traceback
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# ==========================================================
# Forecast generation (Hybrid: Exponential Smoothing + Prophet + LightGBM)
# ==========================================================
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    if "REVENUE" in df.columns and "AMOUNT" not in df.columns:
        df.rename(columns={"REVENUE": "AMOUNT"}, inplace=True)

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {required}")

    for col in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df["Revenue"] = df["AMOUNT"].fillna(df["AMOUNT"].mean())
    df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date")

    if df.empty:
        raise ValueError("No valid data found in Excel file.")

    # ===================== Split data ======================
    train_df = df.iloc[:-30]
    test_df = df.iloc[-30:]

    # ===================== Exponential Smoothing ======================
    es_model = ExponentialSmoothing(
        train_df["Revenue"], trend="add", seasonal=None
    ).fit()
    es_pred = es_model.forecast(30)

    # ===================== Prophet Model ======================
    prophet_df = train_df[["Date", "Revenue"]].rename(columns={"Date": "ds", "Revenue": "y"})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_df)

    future_prophet = prophet.make_future_dataframe(periods=30)
    forecast_prophet = prophet.predict(future_prophet)
    prophet_pred = forecast_prophet["yhat"].tail(30).values

    # ===================== LightGBM Model ======================
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    X = df[["YEAR", "MONTH", "DAY", "DayOfWeek"]]
    y = df["Revenue"]

    model_lgb = lgb.LGBMRegressor(objective="regression", n_estimators=150)
    model_lgb.fit(X[:-30], y[:-30])
    lgb_pred = model_lgb.predict(X[-30:])

    # ===================== Hybrid average ======================
    hybrid_pred = (es_pred + prophet_pred + lgb_pred) / 3
    test_df["Forecast"] = hybrid_pred

    # ===================== Calculate Metrics ======================
    mae = mean_absolute_error(test_df["Revenue"], hybrid_pred)
    rmse = np.sqrt(mean_squared_error(test_df["Revenue"], hybrid_pred))

    # Safe MAPE (avoids division by zero)
    actual = np.array(test_df["Revenue"])
    predicted = np.array(hybrid_pred)
    non_zero_actuals = actual != 0

    if np.any(non_zero_actuals):
        mape = np.mean(np.abs((actual[non_zero_actuals] - predicted[non_zero_actuals]) / actual[non_zero_actuals])) * 100
    else:
        mape = 0

    # ===================== Forecast future 30 days ======================
    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "YEAR": [d.year for d in future_dates],
        "MONTH": [d.month for d in future_dates],
        "DAY": [d.day for d in future_dates],
        "DayOfWeek": [d.dayofweek for d in future_dates],
    })

    future_df["Forecasted_Revenue"] = model_lgb.predict(
        future_df[["YEAR", "MONTH", "DAY", "DayOfWeek"]]
    )

    # Sundays closed
    future_df.loc[future_df["DayOfWeek"] == 6, "Forecasted_Revenue"] = 0

    total_forecast = future_df["Forecasted_Revenue"].sum()

    result = {
        "daily_forecast": {
            str(d.date()): round(r, 2)
            for d, r in zip(future_df["Date"], future_df["Forecasted_Revenue"])
        },
        "total_forecast": round(total_forecast, 2),
        "metrics": {
            "MAE": float(round(mae, 2)),
            "RMSE": float(round(rmse, 2)),
            "MAPE": float(round(mape, 2))
        }
    }

    return result


# ==========================================================
# API ROUTES
# ==========================================================
@app.route("/")
def home():
    return jsonify({"message": "Hybrid Revenue Forecast API active"})


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


@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        df = pd.DataFrame(list(result["daily_forecast"].items()), columns=["Date", "Forecasted_Revenue"])

        df.loc[len(df)] = ["Total (₱)", result["total_forecast"]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast_30_Days")
        output.seek(0)

        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
