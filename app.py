from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math, traceback
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"

def safe_float(x):
    """Ensure value is always JSON serializable float"""
    try:
        if pd.isna(x):
            return 0.0
        val = float(x)
        return val if math.isfinite(val) else 0.0
    except:
        return 0.0


def generate_forecast():
    # Load data
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    df["REVENUE"] = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)

    # ---------- STEP 1: Exponential Smoothing ----------
    es_model = ExponentialSmoothing(df["REVENUE"], trend="add")
    es_fit = es_model.fit()
    es_forecast = es_fit.forecast(30).astype(float)

    # ---------- STEP 2: Prophet ----------
    prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_df)
    prophet_future = prophet.make_future_dataframe(periods=30)
    prophet_pred = prophet.predict(prophet_future)
    prophet_forecast = prophet_pred.tail(30)["yhat"].astype(float).values

    # ---------- STEP 3: LightGBM residual correction ----------
    # Create features
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["month"] = df["DATE"].dt.month

    # Hybrid base (recursive)
    df["Hybrid_forecast"] = (
        0.3 * df["REVENUE"]
        + 0.3 * df["REVENUE"].shift(1).fillna(0)
        + 0.4 * df["REVENUE"].shift(2).fillna(0)
    )

    df["y_smooth"] = df["REVENUE"].rolling(3, min_periods=1).mean()
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]

    features = ["day_of_week", "is_weekend", "month"]
    X_train = df[features]
    y_train = df["Residual"]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    lgb_model.fit(X_train, y_train)

    # ---------- STEP 4: Forecast next 30 days ----------
    future_dates = pd.date_range(df["DATE"].max() + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        "DATE": future_dates,
        "day_of_week": future_dates.dayofweek,
        "is_weekend": (future_dates.dayofweek.isin([5,6])).astype(int),
        "month": future_dates.month
    })

    # LightGBM residuals for future
    future_resid = lgb_model.predict(future_df[features]).astype(float)

    # Combine hybrid base from ES + Prophet + residual correction
    hybrid_future = 0.3 * es_forecast + 0.3 * prophet_forecast + 0.4 * es_forecast
    corrected_future = hybrid_future + future_resid

    # Set Sundays to zero (clinic closed)
    for i, d in enumerate(future_dates):
        if d.dayofweek == 6:
            corrected_future[i] = 0.0

    # Convert to plain floats
    corrected_future = [safe_float(x) for x in corrected_future]

    # ---------- STEP 5: Metrics ----------
    mae = mean_absolute_error(df["y_smooth"], df["Hybrid_forecast"])
    mae_monthly = mae * 30
    total_forecast = round(float(np.sum(corrected_future)), 2)

    print(f"✅ Forecast generated: total ₱{total_forecast:,.2f}, MAE ₱{mae:,.2f}")

    # ---------- STEP 6: Return to dashboard ----------
    chart_data = [
        {"date": str(d.date()), "revenue": safe_float(v), "type": "forecast"}
        for d, v in zip(future_dates, corrected_future)
    ]

    return {
        "status": "success",
        "data": {
            "chart_data": chart_data,
            "total_forecast": total_forecast,
            "mae_daily": round(mae, 2),
            "mae_monthly": round(mae_monthly, 2)
        }
    }


@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_api():
    try:
        result = generate_forecast()
        return jsonify(result), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
