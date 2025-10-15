from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, traceback, math
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"

def safe_float(x):
    try:
        if pd.isna(x): return 0.0
        val = float(x)
        return val if math.isfinite(val) else 0.0
    except:
        return 0.0

def generate_forecast():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df["REVENUE"] = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)

    print("✅ Loaded rows:", len(df))
    print("Sample revenue:", df["REVENUE"].head().tolist())

    # Exponential smoothing
    try:
        es_model = ExponentialSmoothing(df["REVENUE"], trend="add")
        es_fit = es_model.fit()
        es_forecast = es_fit.forecast(30)
        print("🔹 ES forecast sample:", es_forecast[:5].tolist())
    except Exception as e:
        print("❌ ES failed:", e)
        es_forecast = np.zeros(30)

    # Prophet
    try:
        prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
        prophet = Prophet(daily_seasonality=True)
        prophet.fit(prophet_df)
        prophet_future = prophet.make_future_dataframe(periods=30)
        prophet_pred = prophet.predict(prophet_future)
        prophet_forecast = prophet_pred.tail(30)["yhat"].values
        print("🔹 Prophet forecast sample:", prophet_forecast[:5])
    except Exception as e:
        print("❌ Prophet failed:", e)
        prophet_forecast = np.zeros(30)

    # LightGBM
    try:
        features = ["day_of_week", "is_weekend", "month"]
        df["day_of_week"] = df["DATE"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["month"] = df["DATE"].dt.month

        X = df[features]
        y = df["REVENUE"]
        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
        lgb_model.fit(X, y)

        future_dates = pd.date_range(df["DATE"].max() + pd.Timedelta(days=1), periods=30)
        future_df = pd.DataFrame({
            "DATE": future_dates,
            "day_of_week": future_dates.dayofweek,
            "is_weekend": (future_dates.dayofweek.isin([5,6])).astype(int),
            "month": future_dates.month
        })
        lgb_future = lgb_model.predict(future_df[features])
        print("🔹 LightGBM forecast sample:", lgb_future[:5])
    except Exception as e:
        print("❌ LightGBM failed:", e)
        lgb_future = np.zeros(30)

    # Hybrid forecast combination
    try:
        hybrid_future = 0.3 * np.array(es_forecast) + 0.3 * np.array(prophet_forecast) + 0.4 * np.array(lgb_future)
        print("🔹 Hybrid forecast sample:", hybrid_future[:5])
    except Exception as e:
        print("❌ Hybrid failed:", e)
        hybrid_future = np.zeros(30)

    # Residual correction
    try:
        df["y_smooth"] = df["REVENUE"].rolling(3, min_periods=1).mean()
        df["Hybrid_forecast"] = 0.3 * df["REVENUE"] + 0.3 * df["REVENUE"].shift(1).fillna(0) + 0.4 * df["REVENUE"]
        df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]
        resid_features = ["day_of_week", "is_weekend", "month"]
        resid_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        resid_model.fit(df[resid_features], df["Residual"])
        future_resid = resid_model.predict(future_df[resid_features])
        corrected_future = hybrid_future + future_resid
        print("🔹 Corrected forecast sample:", corrected_future[:5])
    except Exception as e:
        print("❌ Residual correction failed:", e)
        corrected_future = hybrid_future

    # Zero Sundays only
    for i, d in enumerate(future_dates):
        if d.dayofweek == 6:
            corrected_future[i] = 0

    print("🔹 After Sunday correction:", corrected_future[:10])

    mae = mean_absolute_error(df["y_smooth"], df["Hybrid_forecast"])
    total_forecast = round(np.sum(corrected_future), 2)

    print(f"✅ Total forecast sum: {total_forecast:,.2f} | MAE: {mae:,.2f}")

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
