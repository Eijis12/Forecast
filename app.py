from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
import os
import math
from datetime import datetime

# Forecasting libs
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"
HOLIDAYS_CSV = "holidays.csv"


# --------------------------
# Helpers
# --------------------------
def safe_float(x, fallback=0.0):
    try:
        if x is None:
            return float(fallback)
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        f = float(x)
        if math.isfinite(f):
            return f
        return float(fallback)
    except Exception:
        return float(fallback)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else 0.0
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    return obj


def is_last_day_of_month(ts: pd.Timestamp):
    nxt = ts + pd.Timedelta(days=1)
    return nxt.month != ts.month


def load_holidays():
    if os.path.exists(HOLIDAYS_CSV):
        try:
            h = pd.read_csv(HOLIDAYS_CSV)
            col = None
            for c in h.columns:
                if c.strip().upper() in ("DATE", "DATES"):
                    col = c
                    break
            if col is None:
                col = h.columns[0]
            hdates = pd.to_datetime(h[col], errors="coerce").dropna().dt.normalize().unique()
            return set(pd.to_datetime(hdates).date)
        except Exception:
            return set()
    return set()


# --------------------------
# Forecasting (Dynamic Variation)
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    holidays_set = load_holidays()

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    if "REVENUE" in df.columns:
        df.rename(columns={"REVENUE": "AMOUNT"}, inplace=True)

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if "DATE" in df.columns:
        required.discard("YEAR")
        required.discard("MONTH")
        required.discard("DAY")

    if not required.issubset(set(df.columns)):
        if not ("DATE" in df.columns and "AMOUNT" in df.columns):
            raise ValueError(f"Excel must contain columns: YEAR, MONTH, DAY, AMOUNT (or DATE + AMOUNT). Found: {df.columns.tolist()}")

    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")

    if "DATE" in df.columns:
        df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")
        if df["DATE_PARSED"].notna().any():
            df["DATE"] = df["DATE_PARSED"]
    if "DATE" not in df.columns or df["DATE"].isna().all():
        df["YEAR"] = pd.to_numeric(df.get("YEAR", pd.Series()), errors="coerce")
        df["MONTH"] = pd.to_numeric(df.get("MONTH", pd.Series()), errors="coerce")
        df["DAY"] = pd.to_numeric(df.get("DAY", pd.Series()), errors="coerce")
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")

    df = df.dropna(subset=["DATE", "AMOUNT"]).copy()
    df = df.sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data rows found after cleaning.")

    df["REVENUE"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0).astype(float)
    df["y_smooth"] = df["REVENUE"].astype(float)

    # --------------------------
    # Feature Engineering
    # --------------------------
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["day"] = df["DATE"].dt.day
    df["is_payday"] = df["DATE"].apply(lambda d: 1 if (d.day == 15 or is_last_day_of_month(d)) else 0)
    df["is_holiday"] = df["DATE"].dt.date.apply(lambda d: 1 if d in holidays_set else 0)

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["REVENUE"].shift(lag).fillna(0.0)
    df["roll_mean_3"] = df["REVENUE"].rolling(3, min_periods=1).mean().fillna(0.0)
    df["roll_std_7"] = df["REVENUE"].rolling(7, min_periods=1).std().fillna(0.0)
    df["is_month_end"] = df["DATE"].dt.is_month_end.astype(int)

    # --------------------------
    # Model Training
    # --------------------------
    features = [
        "day_of_week", "is_weekend", "is_payday", "month", "is_holiday",
        "lag_1", "lag_2", "lag_3", "lag_7", "roll_mean_3", "roll_std_7", "is_month_end"
    ]

    # LightGBM Model
    try:
        X = df[features]
        y = df["y_smooth"]
        lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
        lgb_model.fit(X, y)
        lgb_insample = lgb_model.predict(X)
    except Exception:
        lgb_model = None
        lgb_insample = np.repeat(df["y_smooth"].mean(), len(df))

    # Prophet Model
    try:
        prophet_df = df[["DATE", "y_smooth"]].rename(columns={"DATE": "ds", "y_smooth": "y"})
        prophet_model = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.25,
            seasonality_prior_scale=12.0,
            seasonality_mode="add"
        )
        prophet_model.fit(prophet_df)
        future_prophet = prophet_model.make_future_dataframe(periods=30, freq="D")
        prophet_pred_full = prophet_model.predict(future_prophet)
        prophet_insample = prophet_pred_full.loc[: len(prophet_df)-1, "yhat"].values
        prophet_future = prophet_pred_full.tail(30)["yhat"].values
    except Exception:
        prophet_insample = np.repeat(df["y_smooth"].mean(), len(df))
        prophet_future = np.repeat(df["y_smooth"].mean(), 30)

    # Exponential Smoothing Model
    try:
        es_model = ExponentialSmoothing(df["y_smooth"], trend="add", seasonal="add", seasonal_periods=7)
        es_fit = es_model.fit(optimized=True)
        es_insample = es_fit.fittedvalues
        es_forecast_future = es_fit.forecast(30)
    except Exception:
        es_insample = np.repeat(df["y_smooth"].mean(), len(df))
        es_forecast_future = np.repeat(df["y_smooth"].mean(), 30)

    # --------------------------
    # Future DataFrame
    # --------------------------
    future_start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=30, freq="D").to_pydatetime().tolist()
    future_df = pd.DataFrame({"DATE": pd.to_datetime(future_dates)})

    future_df["day_of_week"] = future_df["DATE"].dt.dayofweek
    future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)
    future_df["month"] = future_df["DATE"].dt.month
    future_df["day"] = future_df["DATE"].dt.day
    future_df["is_payday"] = future_df["DATE"].apply(lambda d: 1 if (d.day == 15 or is_last_day_of_month(d)) else 0)
    future_df["is_holiday"] = future_df["DATE"].dt.date.apply(lambda d: 1 if d in holidays_set else 0)
    future_df["is_month_end"] = future_df["DATE"].dt.is_month_end.astype(int)
    for lag in [1, 2, 3, 7, 14]:
        future_df[f"lag_{lag}"] = df["REVENUE"].iloc[-lag] if len(df) >= lag else df["REVENUE"].mean()
    future_df["roll_mean_3"] = df["REVENUE"].tail(3).mean()
    future_df["roll_std_7"] = df["REVENUE"].tail(7).std()

    if lgb_model is not None:
        lgb_future = lgb_model.predict(future_df[features])
    else:
        lgb_future = np.repeat(df["y_smooth"].mean(), 30)

    # --------------------------
    # Hybrid Forecast (Dynamic)
    # --------------------------
    def fix_arr(a, fallback_mean):
        a = np.array(a, dtype=float)
        a = np.where(np.isfinite(a), a, float(fallback_mean))
        return a

    es_arr = fix_arr(es_forecast_future, df["y_smooth"].mean())
    prop_arr = fix_arr(prophet_future, df["y_smooth"].mean())
    lgb_arr = fix_arr(lgb_future, df["y_smooth"].mean())

    # Combine hybrid model with realistic variation
    hybrid_future = 0.25 * es_arr + 0.35 * prop_arr + 0.40 * lgb_arr

    np.random.seed(42)
    variation = np.random.uniform(-0.15, 0.15, size=len(hybrid_future))
    hybrid_future = hybrid_future * (1 + variation)
    hybrid_future = np.clip(hybrid_future, 10000, 22000)

    # Apply Sunday=0
    hybrid_future_clean = []
    for d, val in zip(future_df["DATE"], hybrid_future):
        if int(pd.Timestamp(d).dayofweek) == 6:
            hybrid_future_clean.append(0.0)
        else:
            hybrid_future_clean.append(float(safe_float(val, 0.0)))

    # --------------------------
    # Output
    # --------------------------
    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            "date": str(pd.Timestamp(row["DATE"]).date()),
            "revenue": safe_float(row["REVENUE"], 0.0),
            "type": "historical"
        })
    for d, v in zip(future_df["DATE"], hybrid_future_clean):
        chart_data.append({
            "date": str(pd.Timestamp(d).date()),
            "revenue": safe_float(v, 0.0),
            "type": "forecast"
        })

    daily_forecast = {str(pd.Timestamp(d).date()): safe_float(v, 0.0) for d, v in zip(future_df["DATE"], hybrid_future_clean)}
    total_forecast = float(round(sum(hybrid_future_clean), 2))

    # MAE
    mae_daily = None
    mae_monthly = None
    try:
        actual = np.array(df["y_smooth"].values[-30:]) if len(df) > 0 else np.array([])
        pred = np.array(lgb_insample[-30:])
        n = min(len(actual), len(pred))
        if n > 0:
            mae_val = mean_absolute_error(actual[-n:], pred[-n:])
            mae_daily = float(round(mae_val, 2))
            mae_monthly = float(round(mae_daily * 30.0, 2))
    except Exception:
        pass

    result = {
        "chart_data": chart_data,
        "daily_forecast": sanitize(daily_forecast),
        "total_forecast": total_forecast,
        "mae_daily": mae_daily,
        "mae_monthly": mae_monthly
    }

    return sanitize(result)


# --------------------------
# API Routes
# --------------------------
@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API active"}), 200


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
        return jsonify({"status": "success", "data": []}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        df_out = pd.DataFrame(list(result["daily_forecast"].items()), columns=["Date", "Forecasted_Revenue"])
        df_out.loc[len(df_out)] = ["Total (₱)", result.get("total_forecast", 0.0)]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Forecast_30_Days")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# --------------------------
# Run (dev)
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
