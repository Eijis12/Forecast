from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
import os
import math
from datetime import datetime
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
# Core forecasting function
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
    df["y_smooth"] = df["REVENUE"].rolling(window=7, center=True, min_periods=1).median()
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["DATE"].dt.month
    df["day"] = df["DATE"].dt.day
    df["is_payday"] = df["DATE"].apply(lambda d: 1 if (d.day == 15 or is_last_day_of_month(d)) else 0)
    df["is_holiday"] = df["DATE"].dt.date.apply(lambda d: 1 if d in holidays_set else 0)

    # --- Exponential Smoothing ---
    try:
        es_model = ExponentialSmoothing(df["y_smooth"], trend="add", seasonal=None)
        es_fit = es_model.fit(optimized=True)
        es_insample = es_fit.fittedvalues
    except Exception:
        es_insample = np.repeat(df["y_smooth"].mean(), len(df))

    # --- Prophet ---
    try:
        prophet_df = df[["DATE", "y_smooth"]].rename(columns={"DATE": "ds", "y_smooth": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_df)
        start_dt = pd.Timestamp.today().normalize()
        future_prophet = prophet_model.make_future_dataframe(periods=30, freq="D")
        prophet_pred_full = prophet_model.predict(future_prophet)
        prophet_insample = prophet_pred_full.loc[: len(prophet_df)-1, "yhat"].values
        prophet_future = prophet_pred_full.tail(30)["yhat"].values
    except Exception:
        prophet_insample = np.repeat(df["y_smooth"].mean(), len(df))
        prophet_future = np.repeat(df["y_smooth"].mean(), 30)

    # --- LightGBM ---
    try:
        features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]
        X = df[features]
        y = df["y_smooth"]
        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
        lgb_model.fit(X, y)
        lgb_insample = lgb_model.predict(X)
    except Exception:
        lgb_model = None
        lgb_insample = np.repeat(df["y_smooth"].mean(), len(df))

    # --- Future data ---
    future_start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=30, freq="D").to_pydatetime().tolist()
    future_df = pd.DataFrame({"DATE": pd.to_datetime(future_dates)})
    future_df["day_of_week"] = future_df["DATE"].dt.dayofweek
    future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)
    future_df["month"] = future_df["DATE"].dt.month
    future_df["day"] = future_df["DATE"].dt.day
    future_df["is_payday"] = future_df["DATE"].apply(lambda d: 1 if (d.day == 15 or is_last_day_of_month(d)) else 0)
    future_df["is_holiday"] = future_df["DATE"].dt.date.apply(lambda d: 1 if d in holidays_set else 0)

    if lgb_model is not None:
        try:
            lgb_future = lgb_model.predict(future_df[features])
        except Exception:
            lgb_future = np.repeat(df["y_smooth"].mean(), 30)
    else:
        lgb_future = np.repeat(df["y_smooth"].mean(), 30)

    try:
        es_forecast_future = es_fit.forecast(30)
    except Exception:
        es_forecast_future = np.repeat(df["y_smooth"].mean(), 30)

    es_arr = np.array(es_forecast_future)
    prop_arr = np.array(prophet_future)
    lgb_arr = np.array(lgb_future)
    hybrid_future = 0.30 * es_arr + 0.30 * prop_arr + 0.40 * lgb_arr

    hybrid_future_corrected_clean = []
    for d, val in zip(future_df["DATE"], hybrid_future):
        if int(pd.Timestamp(d).dayofweek) == 6:
            hybrid_future_corrected_clean.append(0.0)
        else:
            hybrid_future_corrected_clean.append(float(safe_float(val, 0.0)))

    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            "date": str(pd.Timestamp(row["DATE"]).date()),
            "revenue": safe_float(row["REVENUE"], 0.0),
            "type": "historical"
        })
    for d, v in zip(future_df["DATE"], hybrid_future_corrected_clean):
        chart_data.append({
            "date": str(pd.Timestamp(d).date()),
            "revenue": safe_float(v, 0.0),
            "type": "forecast"
        })

    daily_forecast = { str(pd.Timestamp(d).date()): safe_float(v, 0.0) for d, v in zip(future_df["DATE"], hybrid_future_corrected_clean) }
    total_forecast = float(round(sum(hybrid_future_corrected_clean), 2))

    # --- ✅ FIXED MAE SECTION ---
    mae_daily = None
    mae_monthly = None
    try:
        actual = np.array(df["y_smooth"].values[-30:])
        pred = np.array(lgb_insample[-30:])
        n = min(len(actual), len(pred))
        if n > 0:
            mae_val = mean_absolute_error(actual[-n:], pred[-n:])
            mae_daily = float(round(mae_val, 2))
            mae_monthly = float(round(mae_daily * 30.0, 2))  # multiplied by 30
        else:
            mae_daily = None
            mae_monthly = None
    except Exception:
        mae_daily = None
        mae_monthly = None

    result = {
        "chart_data": chart_data,
        "daily_forecast": sanitize(daily_forecast),
        "total_forecast": total_forecast,
        "mae_daily": mae_daily,
        "mae_monthly": mae_monthly
    }

    return sanitize(result)


# --------------------------
# API routes
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
