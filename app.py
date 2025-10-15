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
# Utility Functions
# --------------------------
def safe_float(x):
    """Convert safely without killing valid numeric values."""
    try:
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        f = float(x)
        if not math.isfinite(f):
            return 0.0
        return f
    except Exception:
        return 0.0


# --------------------------
# Forecast Function
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError("Excel file not found")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    # Identify revenue column
    if "REVENUE" in df.columns:
        revenue_col = "REVENUE"
    elif "AMOUNT" in df.columns:
        revenue_col = "AMOUNT"
    else:
        raise ValueError("Excel must contain a REVENUE or AMOUNT column")

    # Build DATE column
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")

    df.dropna(subset=["DATE"], inplace=True)
    df = df.sort_values("DATE").reset_index(drop=True)
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")
    df = df.dropna(subset=[revenue_col])

    print(f"✅ Loaded {len(df)} rows | Revenue sample:", df[revenue_col].head(5).tolist())

    # ------------------------------
    # Feature Engineering
    # ------------------------------
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["is_payday"] = df["DATE"].dt.day.isin([15, 30]).astype(int)
    df["is_holiday"] = 0

    # ------------------------------
    # Exponential Smoothing
    # ------------------------------
    es_model = ExponentialSmoothing(df[revenue_col], trend="add")
    es_fit = es_model.fit()
    df["ES_fitted"] = es_fit.fittedvalues
    es_forecast = es_fit.forecast(30)

    # ------------------------------
    # Prophet Forecast
    # ------------------------------
    prophet_df = df.rename(columns={revenue_col: "y", "DATE": "ds"})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    prophet_pred = prophet_model.predict(prophet_model.make_future_dataframe(periods=30))
    df["Prophet_fitted"] = prophet_pred["yhat"].iloc[:len(df)].values
    prophet_forecast = prophet_pred.tail(30)["yhat"].values

    # ------------------------------
    # LightGBM Forecast
    # ------------------------------
    features = ["day_of_week", "is_weekend", "month"]
    X = df[features]
    y = df[revenue_col]
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

    # ------------------------------
    # Hybrid Combination
    # ------------------------------
    df["Hybrid_forecast"] = 0.3 * df["ES_fitted"] + 0.3 * df["Prophet_fitted"] + 0.4 * df[revenue_col]
    future_df["Hybrid_future_forecast"] = (
        0.3 * es_forecast + 0.3 * prophet_forecast + 0.4 * lgb_forecast
    )

    # ------------------------------
    # Residual Correction (Recursive)
    # ------------------------------
    df["y_smooth"] = df[revenue_col].rolling(3, min_periods=1).mean()
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]

    resid_features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
    resid_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    )
    resid_model.fit(df[resid_features], df["Residual"])

    df["Residual_pred"] = resid_model.predict(df[resid_features])
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]

    future_df["Residual_pred"] = resid_model.predict(future_df[resid_features])
    future_df["Hybrid_future_corrected"] = (
        future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]
    )

    # Sundays = 0
    sunday_mask = future_df["day_of_week"] == 6
    future_df.loc[sunday_mask, "Hybrid_future_corrected"] = 0.0

    # ------------------------------
    # Validate Non-Zero Forecasts
    # ------------------------------
    print("🔎 Future Forecast (first 5 values):", future_df["Hybrid_future_corrected"].head(5).tolist())

    # ------------------------------
    # MAE
    # ------------------------------
    actual = df["y_smooth"].values
    predicted = df["Hybrid_corrected"].values
    mae = mean_absolute_error(actual, predicted)
    mae_monthly = round(mae * 30, 2)

    # ------------------------------
    # Response
    # ------------------------------
    daily_forecast = {
        str(d.date()): safe_float(v)
        for d, v in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
    }
    total_forecast = round(sum(daily_forecast.values()), 2)

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
