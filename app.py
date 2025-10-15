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
# Forecast generation (Recursive Hybrid Model)
# ==========================================================
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # --- Load & clean dataset ---
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    required_cols = {"YEAR", "MONTH", "DAY", "REVENUE"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel must have columns: {required_cols}")

    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df = df.dropna(subset=["DATE", "REVENUE"]).sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data found in Excel.")

    # =====================================================
    # Step 1 – Base Exponential Smoothing
    # =====================================================
    try:
        exp_model = ExponentialSmoothing(df["REVENUE"], trend="add", seasonal=None)
        exp_fit = exp_model.fit()
        df["ES_FIT"] = exp_fit.fittedvalues
        exp_forecast = exp_fit.forecast(30)
    except Exception as e:
        print("⚠️ Exponential smoothing failed:", e)
        exp_forecast = np.repeat(df["REVENUE"].mean(), 30)
        df["ES_FIT"] = df["REVENUE"].rolling(3, min_periods=1).mean()

    # =====================================================
    # Step 2 – Prophet Forecast
    # =====================================================
    prophet_df = df.rename(columns={"DATE": "ds", "REVENUE": "y"})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future_prophet = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future_prophet)
    prophet_pred = prophet_forecast.tail(30)["yhat"].values
    df["PROPHET_FIT"] = prophet_model.predict(prophet_df)["yhat"]

    # =====================================================
    # Step 3 – Combine ES + Prophet into Hybrid
    # =====================================================
    df["Hybrid_forecast"] = 0.5 * df["ES_FIT"] + 0.5 * df["PROPHET_FIT"]
    hybrid_future_base = 0.5 * exp_forecast + 0.5 * prophet_pred

    # =====================================================
    # Step 4 – LightGBM residual correction (Recursive)
    # =====================================================
    df["Residual"] = df["REVENUE"] - df["Hybrid_forecast"]

    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["DATE"].dt.month
    df["is_payday"] = df["DAY"].isin([15, 30]).astype(int)
    df["is_holiday"] = ((df["month"] == 12) & (df["DAY"].isin([25, 31]))).astype(int)

    features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
    X_train = df[features]
    y_train = df["Residual"]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgb_model.fit(X_train, y_train)

    df["Residual_pred"] = lgb_model.predict(X_train)
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]

    # =====================================================
    # Step 5 – Forecast next 30 days recursively
    # =====================================================
    start_date = df["DATE"].max()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({"DATE": future_dates})
    future_df["day_of_week"] = future_df["DATE"].dt.dayofweek
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    future_df["month"] = future_df["DATE"].dt.month
    future_df["DAY"] = future_df["DATE"].dt.day
    future_df["is_payday"] = future_df["DAY"].isin([15, 30]).astype(int)
    future_df["is_holiday"] = ((future_df["month"] == 12) & (future_df["DAY"].isin([25, 31]))).astype(int)

    future_df["Hybrid_future_forecast"] = hybrid_future_base
    future_df["Residual_pred"] = lgb_model.predict(future_df[features])
    future_df["Hybrid_future_corrected"] = future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]

    # Sundays closed (revenue = 0)
    future_df.loc[future_df["day_of_week"] == 6, "Hybrid_future_corrected"] = 0

    # =====================================================
    # Step 6 – Metrics (MAE scaled to monthly)
    # =====================================================
    n_eval = min(30, len(df))
    try:
        daily_mae = mean_absolute_error(df["REVENUE"].values[-n_eval:], df["Hybrid_corrected"].values[-n_eval:])
        mae_val = float(round(daily_mae * 30, 2))  # scaled to monthly
    except Exception:
        mae_val = None

    # =====================================================
    # Step 7 – Prepare combined dataset for chart
    # =====================================================
    hist_df = df[["DATE", "REVENUE"]].copy()
    hist_df["TYPE"] = "historical"

    future_chart = pd.DataFrame({
        "DATE": future_df["DATE"],
        "REVENUE": future_df["Hybrid_future_corrected"],
        "TYPE": "forecast"
    })

    combined_df = pd.concat([hist_df, future_chart], ignore_index=True)
    total_forecast = np.sum(future_df["Hybrid_future_corrected"])

    forecast_result = {
        "chart_data": [
            {"date": str(row["DATE"].date()), "revenue": round(row["REVENUE"], 2), "type": row["TYPE"]}
            for _, row in combined_df.iterrows()
        ],
        "daily_forecast": {
            str(d.date()): round(r, 2)
            for d, r in zip(future_df["DATE"], future_df["Hybrid_future_corrected"])
        },
        "total_forecast": round(total_forecast, 2),
        "mae": round(mae_val, 2) if mae_val is not None else None,
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

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
