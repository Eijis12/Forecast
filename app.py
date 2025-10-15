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
# Safe float
# --------------------------
def safe_float(x):
    try:
        if pd.isna(x):
            return 0.0
        f = float(x)
        return f if math.isfinite(f) else 0.0
    except Exception:
        return 0.0


# --------------------------
# Forecast
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    if "REVENUE" not in df.columns:
        raise ValueError("Expected column 'REVENUE'")

    # Date handling
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df.dropna(subset=["DATE"], inplace=True)
    df = df.sort_values("DATE").reset_index(drop=True)
    df["REVENUE"] = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)

    print(f"✅ Loaded {len(df)} rows, sample:", df["REVENUE"].head(3).tolist())

    # Features
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["is_payday"] = df["DATE"].dt.day.isin([15, 30]).astype(int)
    df["is_holiday"] = 0

    # Exponential smoothing
    es_model = ExponentialSmoothing(df["REVENUE"], trend="add")
    es_fit = es_model.fit()
    df["ES_fit"] = es_fit.fittedvalues
    es_forecast = es_fit.forecast(30)

    # Prophet
    prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_df)
    prophet_forecast = prophet.predict(prophet.make_future_dataframe(periods=30))
    df["Prophet_fit"] = prophet_forecast["yhat"].iloc[:len(df)].values
    prophet_future = prophet_forecast.tail(30)["yhat"].values

    # LightGBM (structure learner)
    features = ["day_of_week", "is_weekend", "month"]
    X = df[features]
    y = df["REVENUE"]
    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
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
    lgb_future = lgb_model.predict(future_df[features])

    # Hybrid combination
    df["Hybrid_forecast"] = (
        0.3 * df["ES_fit"] + 0.3 * df["Prophet_fit"] + 0.4 * df["REVENUE"]
    )
    future_df["Hybrid_future_forecast"] = (
        0.3 * es_forecast + 0.3 * prophet_future + 0.4 * lgb_future
    )

    # Residual correction (recursive)
    df["y_smooth"] = df["REVENUE"].rolling(3, min_periods=1).mean()
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]
    resid_features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
    resid_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    resid_model.fit(df[resid_features], df["Residual"])
    df["Residual_pred"] = resid_model.predict(df[resid_features])
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]
    future_df["Residual_pred"] = resid_model.predict(future_df[resid_features])
    future_df["Hybrid_future_corrected"] = (
        future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]
    )

    # Fix Sundays only
    sunday_count = 0
    for i in range(len(future_df)):
        if future_df.loc[i, "day_of_week"] == 6:  # Sunday
            sunday_count += 1
            future_df.loc[i, "Hybrid_future_corrected"] = 0.0
    print(f"📅 Sundays zeroed out: {sunday_count} days")

    # Debug: preview values
    print("🔍 Forecast preview:", future_df["Hybrid_future_corrected"].head(10).tolist())

    # Metrics
    mae = mean_absolute_error(df["y_smooth"], df["Hybrid_corrected"])
    mae_monthly = round(mae * 30, 2)

    # Output
    daily_forecast = {
        str(d.date()): safe_float(v)
        for d, v in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
    }
    total_forecast = round(sum(daily_forecast.values()), 2)

    print(f"✅ Total forecast sum: {total_forecast:,.2f} | MAE(monthly): {mae_monthly:,.2f}")

    chart_data = [
        {"date": str(d.date()), "revenue": safe_float(v), "type": "forecast"}
        for d, v in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
    ]

    return {
        "status": "success",
        "data": {
            "chart_data": chart_data,
            "daily_forecast": daily_forecast,
            "total_forecast": total_forecast,
            "mae_daily": round(mae, 2),
            "mae_monthly": mae_monthly
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
