# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import io
import math
import traceback
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

# Local default Excel (your file)
EXCEL_PATH = "Dental_Revenue_2425.xlsx"
HISTORY_PATH = "forecast_history.xlsx"

# -------------------------
# Utilities
# -------------------------
def safe_float(x, fallback=0.0):
    try:
        if x is None:
            return float(fallback)
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        f = float(x)
        return f if math.isfinite(f) else float(fallback)
    except Exception:
        return float(fallback)

def sanitize(obj):
    # Convert numpy types and remove nan/inf recursively
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

def ensure_history_exists():
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame(columns=["Date", "Forecasted_Revenue", "mae_daily", "mae_monthly"]).to_excel(HISTORY_PATH, index=False)

# -------------------------
# Core: recursive hybrid forecast
# -------------------------
def recursive_hybrid_forecast(df_hist, days_ahead=30):
    """
    df_hist: DataFrame with columns ['DATE' or 'Date' (datetime), 'REVENUE' numeric]
    Returns: future_dates list, future_values list (floats)
    Recursive hybrid:
      - Fit Prophet on history (ds, y)
      - Fit LightGBM on features including lags
      - For each next day: compute prophet yhat and LGBM predict using latest lags, blend
      - Force Sundays to 0
      - Optionally train a residual LGBM on historical residuals and apply to future
    """
    df = df_hist.copy()

    # Normalize column labels
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "Date"})
    if "REVENUE" in df.columns:
        df = df.rename(columns={"REVENUE": "Revenue"})
    if "AMOUNT" in df.columns and "Revenue" not in df.columns:
        df = df.rename(columns={"AMOUNT": "Revenue"})

    if "Date" not in df.columns or "Revenue" not in df.columns:
        raise ValueError("Input DF must contain 'Date' and 'Revenue' columns")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid historical rows found in data")

    # Prepare prophet
    prophet_df = df.rename(columns={"Date": "ds", "Revenue": "y"})[["ds", "y"]]
    prophet_model = None
    try:
        prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        prophet_model.fit(prophet_df)
    except Exception:
        prophet_model = None

    # Prepare features and LGBM
    # Build lags
    df_feat = df.copy()
    df_feat["dayofweek"] = df_feat["Date"].dt.dayofweek  # Mon=0..Sun=6
    df_feat["month"] = df_feat["Date"].dt.month
    # create several lags (1 and 7)
    df_feat["lag1"] = df_feat["Revenue"].shift(1)
    df_feat["lag7"] = df_feat["Revenue"].shift(7)
    df_feat = df_feat.dropna().reset_index(drop=True)

    lgb_model = None
    features = ["dayofweek", "month", "lag1", "lag7"]
    if len(df_feat) >= 20:
        try:
            X = df_feat[features]
            y = df_feat["Revenue"]
            lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
            lgb_model.fit(X, y)
        except Exception:
            lgb_model = None

    # Build residual model on hybrid in-sample residuals (optional)
    resid_model = None
    try:
        if prophet_model is not None and lgb_model is not None and len(df_feat) >= 60:
            # get in-sample prophet predictions on original dates
            insample_prop = prophet_model.predict(prophet_df)["yhat"].values
            # For ES or other baseline we skip and compute hybrid_in_sample = 0.5*prophet + 0.5*lgb_insample
            # Prepare lgb in-sample using same rows that have lag1/lag7 (df_feat)
            lgb_insample_preds = lgb_model.predict(df_feat[features])
            # Align lengths: df_feat corresponds to df[lag7 valid rows]
            hybrid_insample = 0.5 * df_feat["y"].values + 0.5 * lgb_insample_preds  # note: this is a simple approach
            residuals = df_feat["Revenue"].values - hybrid_insample
            resid_X = df_feat[features]
            resid_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
            resid_model.fit(resid_X, residuals)
    except Exception:
        resid_model = None

    # Start recursive loop
    last_df = df.copy().reset_index(drop=True)
    future_dates = []
    future_preds = []

    # starting point for lag values
    for i in range(days_ahead):
        next_date = last_df.loc[len(last_df) - 1, "Date"] + timedelta(days=1)

        weekday = next_date.weekday()  # 0..6
        if weekday == 6:
            # Sunday closed
            next_pred = 0.0
        else:
            # Prophet pred
            prop_val = None
            try:
                if prophet_model is not None:
                    prop_val = prophet_model.predict(pd.DataFrame({"ds": [next_date]}))["yhat"].iloc[0]
            except Exception:
                prop_val = None

            # Build lgb features using most recent rows to get lag1 and lag7
            last_y = float(last_df.loc[len(last_df) - 1, "Revenue"])
            if len(last_df) >= 7:
                lag7 = float(last_df.loc[len(last_df) - 7, "Revenue"])
            else:
                lag7 = last_y

            lgb_val = None
            if lgb_model is not None:
                try:
                    x_row = pd.DataFrame([{
                        "dayofweek": weekday,
                        "month": next_date.month,
                        "lag1": last_y,
                        "lag7": lag7
                    }])
                    lgb_val = lgb_model.predict(x_row[features])[0]
                except Exception:
                    lgb_val = None

            # Blend: prefer both if present, else whichever exists
            if (prop_val is not None) and (lgb_val is not None):
                next_pred = 0.5 * float(prop_val) + 0.5 * float(lgb_val)
            elif prop_val is not None:
                next_pred = float(prop_val)
            elif lgb_val is not None:
                next_pred = float(lgb_val)
            else:
                next_pred = float(last_df["Revenue"].mean())

            # apply residual correction if present
            if resid_model is not None:
                try:
                    resid_feat = pd.DataFrame([{
                        "dayofweek": weekday,
                        "month": next_date.month,
                        "lag1": last_y,
                        "lag7": lag7
                    }])
                    resid_adj = resid_model.predict(resid_feat)[0]
                    next_pred = float(next_pred) + float(resid_adj)
                except Exception:
                    pass

        # sanitize
        next_pred = safe_float(next_pred, 0.0)
        # append
        future_dates.append(next_date)
        future_preds.append(next_pred)

        # add to last_df to update lags for future iterations
        last_df = pd.concat([last_df, pd.DataFrame({"Date": [next_date], "Revenue": [next_pred]})], ignore_index=True)

    return future_dates, future_preds

# -------------------------
# Generate forecast wrapper (handles reading Excel or uploaded file)
# -------------------------
def generate_forecast_obj(excel_file_path=None, uploaded_file=None):
    """
    Returns sanitized dict with keys:
      - chart_data: list(hist + forecast rows)
      - daily_forecast: dict(date_str -> float)
      - total_forecast: float
      - mae_daily: float or None
      - mae_monthly: float or None (mae_daily * 30)
    """
    # Read data (priority: uploaded_file -> excel_file_path -> default EXCEL_PATH)
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    elif excel_file_path is not None and os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
    elif os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
    else:
        raise FileNotFoundError("No Excel dataset found (no upload and default file missing).")

    # Normalize columns
    df.columns = [c.strip().upper() for c in df.columns]

    # Accept REVENUE or AMOUNT column names and a DATE column if present
    if "REVENUE" in df.columns:
        df = df.rename(columns={"REVENUE": "Revenue"})
    if "AMOUNT" in df.columns and "Revenue" not in df.columns:
        df = df.rename(columns={"AMOUNT": "Revenue"})
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "Date"})

    required = {"YEAR", "MONTH", "DAY"}  # we support either Date column or YEAR/MONTH/DAY
    # Build Date if needed
    if "Date" not in df.columns or df["Date"].isna().all():
        if required.issubset(set(df.columns)):
            # coerce numeric then combine
            for col in ["YEAR", "MONTH", "DAY"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
        else:
            # If no date columns at all, try to fail gracefully
            raise ValueError("Excel must include 'Date' column or YEAR/MONTH/DAY columns.")

    # Ensure Revenue numeric
    if "Revenue" not in df.columns:
        # Try AMOUNT fallback already done, else error
        raise ValueError("Excel must contain a 'REVENUE' or 'AMOUNT' column.")

    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows after cleaning the Excel file.")

    # compute validation MAE by holding out last 30 days
    n_hold = min(30, len(df) // 5 if len(df) >= 60 else min(30, len(df) // 4))
    # ensure at least 1
    n_hold = max(1, min(30, n_hold))
    if len(df) > n_hold:
        train_df = df.iloc[:-n_hold].rename(columns={"Date": "Date", "Revenue": "Revenue"})
        valid_df = df.iloc[-n_hold:].rename(columns={"Date": "Date", "Revenue": "Revenue"})
    else:
        train_df = df.copy()
        valid_df = pd.DataFrame(columns=df.columns)  # empty

    # Recursive forecast to validate: forecast n_hold days from train_df
    mae_daily = None
    mae_monthly = None
    try:
        if len(train_df) >= 10 and len(valid_df) >= 1:
            _, preds_val = recursive_hybrid_forecast(train_df[["Date", "Revenue"]], days_ahead=len(valid_df))
            preds_val_arr = np.array([safe_float(x, 0.0) for x in preds_val])
            actuals = np.array([safe_float(x, 0.0) for x in valid_df["Revenue"].values[: len(preds_val_arr)]])
            if len(actuals) == len(preds_val_arr) and len(actuals) > 0:
                mae_daily = float(mean_absolute_error(actuals, preds_val_arr))
                mae_monthly = float(round(mae_daily * 30.0, 2))
    except Exception:
        mae_daily = None
        mae_monthly = None

    # Now generate final 30-day forecast using the full history
    future_days = 30
    fut_dates, fut_preds = recursive_hybrid_forecast(df[["Date", "Revenue"]], days_ahead=future_days)

    # set Sundays to 0 again as safety
    fut_preds = [0.0 if d.weekday() == 6 else safe_float(v, 0.0) for d, v in zip(fut_dates, fut_preds)]

    # Build chart_data: historical rows + forecast rows
    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({"date": str(pd.Timestamp(row["Date"]).date()), "revenue": safe_float(row["Revenue"]) , "type": "historical"})
    for d, v in zip(fut_dates, fut_preds):
        chart_data.append({"date": str(pd.Timestamp(d).date()), "revenue": round(float(v), 2), "type": "forecast"})

    # daily_forecast mapping
    daily_forecast = { str(pd.Timestamp(d).date()): round(float(v), 2) for d, v in zip(fut_dates, fut_preds) }
    total_forecast = round(sum([safe_float(x, 0.0) for x in fut_preds]), 2)

    result = {
        "chart_data": chart_data,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_daily, 2) if mae_daily is not None else None,
        "mae_monthly": round(mae_monthly, 2) if mae_monthly is not None else None
    }

    # Optionally append to history file for download
    try:
        ensure_history_exists()
        hist_df = pd.DataFrame(list(daily_forecast.items()), columns=["Date", "Forecasted_Revenue"])
        hist_df["mae_daily"] = result["mae_daily"]
        hist_df["mae_monthly"] = result["mae_monthly"]
        # append current forecast to history sheet
        existing = pd.read_excel(HISTORY_PATH)
        combined = pd.concat([existing, hist_df], ignore_index=True)
        combined.to_excel(HISTORY_PATH, index=False)
    except Exception:
        # non-fatal
        pass

    return sanitize(result)

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"success","message":"Revenue Forecast API running"}), 200

@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def forecast_route():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status":"ok"}), 200

        # accept either a file upload or fallback to local Excel
        uploaded_file = None
        if "file" in request.files:
            uploaded_file = request.files.get("file")

        result = generate_forecast_obj(excel_file_path=EXCEL_PATH, uploaded_file=uploaded_file)
        return jsonify({"status":"success", "data": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": str(e)}), 500

@app.route("/api/revenue/history", methods=["GET", "OPTIONS"])
def history_route():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status":"ok"}), 200
        # return saved history if available, else empty list
        if os.path.exists(HISTORY_PATH):
            df = pd.read_excel(HISTORY_PATH)
            return jsonify({"status":"success", "data": df.to_dict(orient="records")}), 200
        return jsonify({"status":"success", "data": []}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": str(e)}), 500

@app.route("/api/revenue/download", methods=["GET"])
def download_route():
    try:
        # use the latest computed history file
        if not os.path.exists(HISTORY_PATH):
            return jsonify({"status":"error","message":"No history file available"}), 404
        return send_file(HISTORY_PATH, as_attachment=True, download_name="forecast_history.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": str(e)}), 500

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # debug True for development logs
    app.run(host="0.0.0.0", port=5000, debug=True)
