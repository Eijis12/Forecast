from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io, os, math, traceback
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# --------------------------
# Helper functions
# --------------------------
def safe_float(x, fallback=0.0):
    try:
        f = float(x)
        return f if math.isfinite(f) else fallback
    except:
        return fallback


def clean_revenue(x):
    """Removes peso signs, commas, and converts to float."""
    if isinstance(x, str):
        x = x.replace("₱", "").replace(",", "").strip()
    try:
        val = float(x)
        return val if math.isfinite(val) else 0.0
    except:
        return 0.0


# --------------------------
# Forecast generation
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    if "REVENUE" in df.columns:
        df.rename(columns={"REVENUE": "AMOUNT"}, inplace=True)
    elif "AMOUNT" not in df.columns:
        raise ValueError("Excel file must include 'REVENUE' or 'AMOUNT' column.")

    # Clean revenue values properly (handle ₱ signs, commas)
    df["REVENUE"] = df["AMOUNT"].apply(clean_revenue)

    # Construct DATE column if missing
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")

    df.dropna(subset=["DATE"], inplace=True)
    df = df.sort_values("DATE").reset_index(drop=True)

    if df["REVENUE"].sum() == 0:
        raise ValueError("Revenue column contains no valid numeric values.")

    # Features
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["is_payday"] = df["DATE"].dt.day.isin([15, 30]).astype(int)
    df["is_holiday"] = 0

    # ---------------------------------------------
    # Base Models: Exponential Smoothing + Prophet
    # ---------------------------------------------
    es_model = ExponentialSmoothing(df["REVENUE"], trend="add")
    es_fit = es_model.fit()
    df["ES_fitted"] = es_fit.fittedvalues
    es_forecast = es_fit.forecast(30)

    prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    prophet_pred = prophet_model.predict(prophet_model.make_future_dataframe(periods=30))
    df["Prophet_fitted"] = prophet_pred["yhat"].iloc[:len(df)].values
    prophet_forecast = prophet_pred.tail(30)["yhat"].values

    # LightGBM Base Model
    features = ["day_of_week", "is_weekend", "month"]
    X = df[features]
    y = df["REVENUE"]
    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    lgb_model.fit(X, y)

    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({
        "DATE": future_dates,
        "day_of_week": [d.dayofweek for d in future_dates],
        "is_weekend": [1 if d.dayofweek in [5, 6] else 0 for d in future_dates],
        "month": [d.month for d in future_dates],
        "is_payday": [1 if d.day in [15, 30] else 0 for d in future_dates],
        "is_holiday": [0]*30
    })

    lgb_forecast = lgb_model.predict(future_df[features])

    # Base hybrid
    df["Hybrid_forecast"] = 0.3 * df["ES_fitted"] + 0.3 * df["Prophet_fitted"] + 0.4 * y
    future_df["Hybrid_future_forecast"] = (
        0.3 * es_forecast + 0.3 * prophet_forecast + 0.4 * lgb_forecast
    )

    # ---------------------------------------------
    # Residual Correction Layer (recursive logic)
    # ---------------------------------------------
    df["y_smooth"] = df["REVENUE"].rolling(3, min_periods=1).mean()
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]

    resid_features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
    resid_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    resid_model.fit(df[resid_features], df["Residual"])
    df["Residual_pred"] = resid_model.predict(df[resid_features])
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]

    future_df["Residual_pred"] = resid_model.predict(future_df[resid_features])
    future_df["Hybrid_future_corrected"] = (
        future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]
    )

    # Sundays = 0
    future_df.loc[future_df["day_of_week"] == 6, "Hybrid_future_corrected"] = 0.0

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    actual = df["y_smooth"].values
    predicted = df["Hybrid_corrected"].values
    mae_corr = mean_absolute_error(actual, predicted)
    mae_monthly = round(mae_corr * 30, 2)

    # ---------------------------------------------
    # Build response
    # ---------------------------------------------
    daily_forecast = {
        str(d.date()): safe_float(r)
        for d, r in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
    }

    total_forecast = round(sum(daily_forecast.values()), 2)
    chart_data = [
        {"date": str(d.date()), "revenue": safe_float(r), "type": "forecast"}
        for d, r in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
    ]

    return {
        "daily_forecast": daily_forecast,
        "chart_data": chart_data,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_corr, 2),
        "mae_monthly": mae_monthly
    }


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Recursive Hybrid Forecast running"})


@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_route():
    try:
        result = generate_forecast()
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
