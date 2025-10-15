# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
import os
import math
from sklearn.metrics import mean_absolute_error

# Forecasting libs
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"

# --------------------------
# Utilities
# --------------------------
def safe_float(x, fallback=0.0):
    """Return finite python float or fallback."""
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
    """Recursively convert numpy types to python types and replace non-finite floats."""
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

# --------------------------
# Core forecasting: recursive hybrid + validation MAE
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # Read sheet
    df = pd.read_excel(EXCEL_PATH)
    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Accept columns: DATE | YEAR | MONTH | DAY | REVENUE (your dataset)
    # Some files call revenue "AMOUNT" — handle both
    if "REVENUE" in df.columns and "AMOUNT" not in df.columns:
        df = df.rename(columns={"REVENUE": "AMOUNT"})

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    # allow DATE column as well; still require numeric YEAR/MONTH/DAY/AMOUNT
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel file must have columns: {required}. Found: {df.columns.tolist()}")

    # ensure numeric
    for c in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Use DATE column if present and parseable, otherwise build from YEAR/MONTH/DAY
    if "DATE" in df.columns:
        parsed = pd.to_datetime(df["DATE"], errors="coerce")
        if parsed.notna().any():
            df["DATE"] = parsed
        else:
            df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    else:
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")

    # If any missing dates, forward/backfill a little
    if df["DATE"].isna().any():
        df["DATE"] = df["DATE"].fillna(method="ffill").fillna(method="bfill")

    # revenue numeric
    df["REVENUE_VAL"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    df = df.dropna(subset=["DATE", "REVENUE_VAL"]).copy()
    df = df.sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows after cleaning the Excel file.")

    # Basic features
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df["DOW"] = df["DATE"].dt.dayofweek  # Monday=0 ... Sunday=6
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)

    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows for training (need >= ~10).")

    # Train / validation split (80/20)
    split_idx = max(int(n * 0.8), n - 30)  # ensure at least some validation
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    val_horizon = len(val_df)

    # --- 1) Exponential Smoothing on train ---
    try:
        es_model = ExponentialSmoothing(train_df["REVENUE_VAL"], trend="add", seasonal=None)
        es_fit = es_model.fit(optimized=True)
        # forecast for validation horizon and later for 30-days
        es_val_fore = es_fit.forecast(val_horizon)
    except Exception:
        es_fit = None
        es_val_fore = np.repeat(train_df["REVENUE_VAL"].mean(), val_horizon)

    # --- 2) Prophet on train ---
    try:
        prophet_train = train_df[["DATE", "REVENUE_VAL"]].rename(columns={"DATE": "ds", "REVENUE_VAL": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_train)
        # prepare future frame covering validation horizon and also extra 30 days from end of train
        # first compute dates for validation period (train end + 1 .. val end)
        train_end = train_df["DATE"].iloc[-1]
        val_dates = [train_end + pd.Timedelta(days=i) for i in range(1, val_horizon + 1)]
        future_val = pd.DataFrame({"ds": val_dates})
        prophet_val_pred = prophet_model.predict(future_val)
        prop_val_fore = prophet_val_pred["yhat"].values
    except Exception:
        prophet_model = None
        prop_val_fore = np.repeat(train_df["REVENUE_VAL"].mean(), val_horizon)

    # --- 3) LightGBM on train (simple date features) ---
    lgb_model = None
    try:
        feat_cols = ["YEAR", "MONTH", "DAY", "DOW"]
        X_train = train_df[feat_cols].astype(int)
        y_train = train_df["REVENUE_VAL"].values
        lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=0.05)
        lgb_model.fit(X_train, y_train)
        # prepare validation features
        val_feat = pd.DataFrame({"DATE": val_dates})
        val_feat["YEAR"] = val_feat["DATE"].dt.year
        val_feat["MONTH"] = val_feat["DATE"].dt.month
        val_feat["DAY"] = val_feat["DATE"].dt.day
        val_feat["DOW"] = val_feat["DATE"].dt.dayofweek
        lgb_val_fore = lgb_model.predict(val_feat[feat_cols])
    except Exception:
        lgb_model = None
        lgb_val_fore = np.repeat(train_df["REVENUE_VAL"].mean(), val_horizon)

    # --- 4) Combine forecasts (weighted average) for validation horizon ---
    def fix_arr(a):
        a = np.array(a, dtype=float)
        if np.isnan(a).any():
            a = np.where(np.isfinite(a), a, train_df["REVENUE_VAL"].mean())
        return a

    arr_es = fix_arr(es_val_fore)
    arr_prop = fix_arr(prop_val_fore)
    arr_lgb = fix_arr(lgb_val_fore)

    # weights
    w_es, w_prop, w_lgb = 0.30, 0.30, 0.40
    combined_val = w_es * arr_es + w_prop * arr_prop + w_lgb * arr_lgb

    # --- 5) Residual correction model (train on train residuals) ---
    try:
        # Build in-sample hybrid predictions for train to compute residuals
        # es in-sample fitted values (if available) else mean
        if es_fit is not None:
            try:
                es_insample = np.array(es_fit.fittedvalues)
            except Exception:
                es_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))
        else:
            es_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))
        # prophet in-sample predictions
        if prophet_model is not None:
            try:
                prophet_insample = prophet_model.predict(prophet_train)["yhat"].values
            except Exception:
                prophet_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))
        else:
            prophet_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))

        # lgb in-sample predictions
        if lgb_model is not None:
            try:
                lgb_insample = lgb_model.predict(train_df[feat_cols])
            except Exception:
                lgb_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))
        else:
            lgb_insample = np.repeat(train_df["REVENUE_VAL"].mean(), len(train_df))

        in_sample_hybrid = w_es * es_insample + w_prop * prophet_insample + w_lgb * lgb_insample
        residuals = train_df["REVENUE_VAL"].values - in_sample_hybrid

        # only train residual model if we have enough history (>= 60 rows recommended)
        resid_model = None
        if len(train_df) >= 60:
            resid_feats = ["DOW", "MONTH", "DAY", "IS_WEEKEND"]
            resid_X = train_df[resid_feats]
            resid_y = residuals
            resid_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
            resid_model.fit(resid_X, resid_y)
            # predict residuals on validation dates
            val_resid_feats = val_feat.assign(IS_WEEKEND=val_feat["DATE"].dt.dayofweek.isin([5,6]).astype(int))
            resid_val_pred = resid_model.predict(val_resid_feats[resid_feats])
            combined_val = combined_val + resid_val_pred
        # else skip residual correction
    except Exception:
        # ignore residual correction failures
        pass

    # --- 6) apply Sunday rule (Sunday=6 -> revenue 0) for validation
    combined_val_corrected = []
    for d, v in zip(val_dates, combined_val):
        if d.dayofweek == 6:
            combined_val_corrected.append(0.0)
        else:
            combined_val_corrected.append(float(safe_float(v, 0.0)))
    combined_val_corrected = np.array(combined_val_corrected, dtype=float)

    # Compute validation MAE comparing combined_val_corrected vs actual val_df REVENUE_VAL
    mae_val = None
    mae_monthly = None
    try:
        actual_val = np.array(val_df["REVENUE_VAL"].values, dtype=float)
        if len(actual_val) > 0:
            # align lengths (should be same)
            m = min(len(actual_val), len(combined_val_corrected))
            if m > 0:
                mae_val = float(mean_absolute_error(actual_val[:m], combined_val_corrected[:m]))
                mae_monthly = round(mae_val * 30.0, 2)
    except Exception:
        mae_val = None
        mae_monthly = None

    # ------------------------------------------------------------------
    # Now produce final recursive 30-day forecast starting from today
    # ------------------------------------------------------------------
    H = 30
    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, H+1)]

    # ES forecast for H days: fit ES on full history (train+val)
    try:
        full_es = ExponentialSmoothing(df["REVENUE_VAL"], trend="add", seasonal=None).fit(optimized=True)
        es_future = full_es.forecast(H)
    except Exception:
        es_future = np.repeat(df["REVENUE_VAL"].mean(), H)

    # Prophet fit on full history and future H days
    try:
        full_prophet_df = df[["DATE", "REVENUE_VAL"]].rename(columns={"DATE": "ds", "REVENUE_VAL": "y"})
        full_prophet = Prophet(daily_seasonality=True)
        full_prophet.fit(full_prophet_df)
        future_prop = full_prophet.make_future_dataframe(periods=H)
        future_prop_pred = full_prophet.predict(future_prop)
        prop_future = future_prop_pred.tail(H)["yhat"].values
    except Exception:
        prop_future = np.repeat(df["REVENUE_VAL"].mean(), H)

    # LGB predict on future date features
    try:
        future_df = pd.DataFrame({"DATE": future_dates})
        future_df["YEAR"] = future_df["DATE"].dt.year
        future_df["MONTH"] = future_df["DATE"].dt.month
        future_df["DAY"] = future_df["DATE"].dt.day
        future_df["DOW"] = future_df["DATE"].dt.dayofweek
        if lgb_model is not None:
            lgb_future = lgb_model.predict(future_df[["YEAR","MONTH","DAY","DOW"]])
        else:
            lgb_future = np.repeat(df["REVENUE_VAL"].mean(), H)
    except Exception:
        lgb_future = np.repeat(df["REVENUE_VAL"].mean(), H)

    # combine future arrays
    arr_es_f = fix_arr(es_future)
    arr_prop_f = fix_arr(prop_future)
    arr_lgb_f = fix_arr(lgb_future)
    combined_future = w_es * arr_es_f + w_prop * arr_prop_f + w_lgb * arr_lgb_f

    # residual correction trained earlier may be available (resid_model variable)
    try:
        if 'resid_model' in locals() and resid_model is not None:
            future_pred_feats = future_df.assign(IS_WEEKEND=future_df["DOW"].isin([5,6]).astype(int))
            resid_future_pred = resid_model.predict(future_pred_feats[["DOW","MONTH","DAY","IS_WEEKEND"]])
            combined_future = np.array(combined_future) + resid_future_pred
    except Exception:
        pass

    # apply Sunday rule to final 30-day forecast
    combined_future_corrected = []
    for d, v in zip(future_dates, combined_future):
        if d.dayofweek == 6:
            combined_future_corrected.append(0.0)
        else:
            combined_future_corrected.append(float(safe_float(v, 0.0)))

    # Build chart_data (history + future)
    chart_rows = []
    for _, r in df.iterrows():
        chart_rows.append({
            "date": str(r["DATE"].date()),
            "revenue": safe_float(r["REVENUE_VAL"], 0.0),
            "type": "historical"
        })
    for d, v in zip(future_dates, combined_future_corrected):
        chart_rows.append({
            "date": str(pd.Timestamp(d).date()),
            "revenue": safe_float(v, 0.0),
            "type": "forecast"
        })

    # daily_forecast mapping
    daily_forecast = { str(pd.Timestamp(d).date()): safe_float(v, 0.0) for d, v in zip(future_dates, combined_future_corrected) }

    total_forecast = float(round(sum(combined_future_corrected), 2))

    result = {
        "chart_data": chart_rows,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        # return validation MAE (daily) and monthly (30x) for display
        "mae_validation_daily": round(float(mae_val), 2) if mae_val is not None else None,
        "mae_validation_monthly": round(float(mae_monthly), 2) if mae_monthly is not None else None,
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

# --------------------------
# Run (development mode)
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
