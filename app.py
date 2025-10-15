# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
import os
import math

# Forecasting libraries
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# --------------------------
# Helpers
# --------------------------
def safe_float(x, fallback=0.0):
    """Convert numpy/pandas numbers to float and handle NaN/inf."""
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


def ensure_list_of_floats(arr):
    return [safe_float(v, 0.0) for v in list(arr)]


# --------------------------
# Recursive Hybrid Forecast
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    if "REVENUE" in df.columns:
        df = df.rename(columns={"REVENUE": "AMOUNT"})

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel file must have columns: {required}. Found: {df.columns.tolist()}")

    for col in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "DATE" in df.columns:
        df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")
        if df["DATE_PARSED"].notna().any():
            df["DATE"] = df["DATE_PARSED"]
    if "DATE" not in df.columns or df["DATE"].isna().all():
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df["DATE"] = df["DATE"].fillna(method="ffill").fillna(method="bfill")

    df["REVENUE_VAL"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    df = df.dropna(subset=["DATE", "REVENUE_VAL"]).sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data rows found after cleaning.")

    df["DOW"] = df["DATE"].dt.dayofweek
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)

    # --- 1) Exponential Smoothing ---
    try:
        es_model = ExponentialSmoothing(df["REVENUE_VAL"], trend="add", seasonal=None)
        es_fit = es_model.fit(optimized=True)
        es_forecast = es_fit.forecast(30)
        es_fitted = es_fit.fittedvalues
    except Exception:
        es_forecast = pd.Series(np.repeat(df["REVENUE_VAL"].mean(), 30))
        es_fitted = pd.Series(np.repeat(df["REVENUE_VAL"].mean(), len(df)))

    # --- 2) Prophet ---
    try:
        prophet_df = df[["DATE", "REVENUE_VAL"]].rename(columns={"DATE": "ds", "REVENUE_VAL": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_df)
        future_prophet = prophet_model.make_future_dataframe(periods=30)
        prophet_pred = prophet_model.predict(future_prophet)
        prophet_forecast = prophet_pred.tail(30)["yhat"].values
    except Exception:
        prophet_forecast = np.repeat(df["REVENUE_VAL"].mean(), 30)

    # --- 3) LightGBM Recursive Forecast ---
    try:
        lgb_features = ["YEAR", "MONTH", "DAY", "DOW"]
        X = df[lgb_features]
        y = df["REVENUE_VAL"]
        lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=0.05)
        lgb_model.fit(X, y)

        start_date = df["DATE"].max()
        future_dates = []
        preds = []

        last_df = df.copy()
        for i in range(1, 31):
            next_date = start_date + pd.Timedelta(days=i)
            features = {
                "YEAR": next_date.year,
                "MONTH": next_date.month,
                "DAY": next_date.day,
                "DOW": next_date.dayofweek
            }
            X_next = pd.DataFrame([features])
            y_pred = lgb_model.predict(X_next)[0]
            preds.append(y_pred)
            new_row = {"DATE": next_date, "YEAR": next_date.year, "MONTH": next_date.month,
                       "DAY": next_date.day, "DOW": next_date.dayofweek, "REVENUE_VAL": y_pred}
            last_df = pd.concat([last_df, pd.DataFrame([new_row])], ignore_index=True)
            start_date = next_date

        lgb_forecast = np.array(preds)
        future_dates = last_df.tail(30)["DATE"].values
    except Exception:
        future_dates = [pd.Timestamp.today().normalize() + pd.Timedelta(days=i) for i in range(1, 31)]
        lgb_forecast = np.repeat(df["REVENUE_VAL"].mean(), 30)

    # --- 4) Combine ---
    def fix_array(arr):
        arr = np.array(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, df["REVENUE_VAL"].mean())
        return arr

    exp_arr = fix_array(es_forecast)
    prop_arr = fix_array(prophet_forecast)
    lgb_arr = fix_array(lgb_forecast)
    combined = 0.3 * exp_arr + 0.3 * prop_arr + 0.4 * lgb_arr

    # --- 5) Sunday closed (set revenue=0) ---
    combined_corrected = []
    for d, val in zip(future_dates, combined):
        dow = pd.Timestamp(d).dayofweek
        combined_corrected.append(0.0 if dow == 6 else safe_float(val))

    # --- 6) Recursive forecast chart data ---
    chart_rows = [{"date": str(r["DATE"].date()), "revenue": safe_float(r["REVENUE_VAL"]), "type": "historical"} for _, r in df.iterrows()]
    for d, v in zip(future_dates, combined_corrected):
        chart_rows.append({"date": str(pd.Timestamp(d).date()), "revenue": safe_float(v), "type": "forecast"})

    daily_forecast = {str(pd.Timestamp(d).date()): safe_float(v) for d, v in zip(future_dates, combined_corrected)}

    # --- 7) MAE (monthly scaled) ---
    mae_val, mae_monthly = None, None
    try:
        insample_pred = np.array(es_fitted[-30:]) if len(es_fitted) >= 1 else np.repeat(df["REVENUE_VAL"].mean(), min(30, len(df)))
        actual_last = np.array(df["REVENUE_VAL"].values[-30:]) if len(df) >= 1 else np.array([])
        n = min(len(actual_last), len(insample_pred))
        if n > 0:
            mae_val = mean_absolute_error(actual_last[-n:], insample_pred[-n:])
            mae_monthly = round(mae_val * 30.0, 2)
    except Exception:
        mae_val, mae_monthly = None, None

    total_forecast = float(round(sum(combined_corrected), 2))

    result = {
        "chart_data": chart_rows,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_val, 2) if mae_val is not None else None,
        "mae_monthly": mae_monthly
    }

    return result


# --------------------------
# Routes
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
        tb = traceback.format_exc()
        print(tb)
        msg = str(e)
        return jsonify({"status": "error", "message": msg}), 500


@app.route("/api/revenue/history", methods=["GET", "OPTIONS"])
def get_history():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        return jsonify({"status": "success", "data": []}), 200
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
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
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"status": "error", "message": str(e)}), 500


# --------------------------
# Run locally
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
