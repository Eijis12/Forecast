from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import os
import traceback
import math
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"

# ------------------------------
# Helper functions
# ------------------------------
def safe_float(x, fallback=0.0):
    try:
        if pd.isna(x) or not np.isfinite(x):
            return fallback
        return float(x)
    except:
        return fallback

# ------------------------------
# Forecast Generation
# ------------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    if "REVENUE" not in df.columns:
        raise ValueError("Excel must contain column 'REVENUE'")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "REVENUE"]).sort_values("DATE")
    df["REVENUE"] = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)

    # Feature engineering
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["is_payday"] = df["DATE"].dt.day.isin([15, 30]).astype(int)
    df["is_holiday"] = 0  # placeholder (could load from holiday list)

    # ---------------------------
    # Hybrid baseline forecast
    # ---------------------------
    # 1. Exponential Smoothing
    try:
        es_model = ExponentialSmoothing(df["REVENUE"], trend="add", seasonal=None)
        es_fit = es_model.fit()
        df["ES_fitted"] = es_fit.fittedvalues
        es_forecast = es_fit.forecast(30)
    except:
        df["ES_fitted"] = df["REVENUE"].mean()
        es_forecast = np.repeat(df["REVENUE"].mean(), 30)

    # 2. Prophet
    prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future_prophet = prophet_model.make_future_dataframe(periods=30)
    prophet_pred = prophet_model.predict(future_prophet)
    df["Prophet_fitted"] = prophet_pred["yhat"].iloc[:len(df)].values
    prophet_forecast = prophet_pred.tail(30)["yhat"].values

    # 3. LightGBM (baseline hybrid feature learner)
    df["DAY"] = df["DATE"].dt.day
    features = ["day_of_week", "is_weekend", "month", "DAY"]
    X = df[features]
    y = df["REVENUE"]
    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    lgb_model.fit(X, y)

    # Future feature setup
    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({
        "DATE": future_dates,
        "day_of_week": [d.dayofweek for d in future_dates],
        "is_weekend": [1 if d.dayofweek in [5,6] else 0 for d in future_dates],
        "month": [d.month for d in future_dates],
        "DAY": [d.day for d in future_dates],
        "is_payday": [1 if d.day in [15,30] else 0 for d in future_dates],
        "is_holiday": [0]*30
    })
    lgb_forecast = lgb_model.predict(future_df[features])

    # 4. Combine hybrid forecasts
    df["Hybrid_forecast"] = (0.3*df["ES_fitted"] + 0.3*df["Prophet_fitted"] + 0.4*y)
    hybrid_future_forecast = (0.3*es_forecast + 0.3*prophet_forecast + 0.4*lgb_forecast)
    future_df["Hybrid_future_forecast"] = hybrid_future_forecast

    # ---------------------------
    # Residual correction (recursive)
    # ---------------------------
    df["y_smooth"] = df["REVENUE"].rolling(3, min_periods=1).mean()
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]

    features_resid = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
    X_train = df[features_resid]
    y_train = df["Residual"]

    resid_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    resid_model.fit(X_train, y_train)
    df["Residual_pred"] = resid_model.predict(X_train)
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]

    # Predict residuals for future
    future_df["Residual_pred"] = resid_model.predict(future_df[features_resid])
    future_df["Hybrid_future_corrected"] = future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]

    # Apply Sunday = 0 rule
    future_df.loc[future_df["day_of_week"] == 6, "Hybrid_future_corrected"] = 0.0

    # ---------------------------
    # Evaluation metrics (MAE)
    # ---------------------------
    actual = df["y_smooth"].values
    predicted = df["Hybrid_corrected"].values
    mae_corr = mean_absolute_error(actual, predicted)
    mae_monthly = round(mae_corr * 30, 2)

    # ---------------------------
    # Prepare output
    # ---------------------------
    combined_df = pd.concat([
        df[["DATE", "REVENUE"]].assign(Type="historical").rename(columns={"REVENUE": "Revenue"}),
        future_df[["DATE", "Hybrid_future_corrected"]].assign(Type="forecast").rename(columns={"Hybrid_future_corrected": "Revenue"})
    ])

    chart_data = [
        {"date": str(row["DATE"].date()), "revenue": safe_float(row["Revenue"]), "type": row["Type"]}
        for _, row in combined_df.iterrows()
    ]

    daily_forecast = {str(d.date()): safe_float(r) for d, r in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])}

    total_forecast = round(sum(daily_forecast.values()), 2)

    return {
        "chart_data": chart_data,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_corr, 2),
        "mae_monthly": mae_monthly
    }

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return jsonify({"message": "Recursive Hybrid Forecast API active"}), 200

@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_revenue():
    try:
        result = generate_forecast()
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        df_out = pd.DataFrame(list(result["daily_forecast"].items()), columns=["Date", "Forecasted_Revenue"])
        df_out.loc[len(df_out)] = ["Total (₱)", result["total_forecast"]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="RecursiveForecast_30_Days")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="RecursiveForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
