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
    """Convert numpy and pandas numbers to Python float and convert NaN/inf to fallback."""
    try:
        if x is None:
            return float(fallback)
        # if it's a numpy scalar
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        # if it's pandas Timestamp or similar, not relevant here
        f = float(x)
        if math.isfinite(f):
            return f
        return float(fallback)
    except Exception:
        return float(fallback)


def ensure_list_of_floats(arr):
    return [safe_float(v, 0.0) for v in list(arr)]


# --------------------------
# Core forecasting function
# --------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # Read excel
    df = pd.read_excel(EXCEL_PATH)

    # Normalize column names (upper, stripped)
    df.columns = [c.strip().upper() for c in df.columns]

    # Accept either "REVENUE" or "AMOUNT" as revenue column. Prefer REVENUE if present.
    if "REVENUE" in df.columns:
        df = df.rename(columns={"REVENUE": "AMOUNT"})
    # Now require YEAR, MONTH, DAY, AMOUNT
    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel file must have columns: {required}. Found: {df.columns.tolist()}")

    # Convert numeric columns safely
    for col in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create date column (if DATE present, prefer it)
    if "DATE" in df.columns:
        # try parsing DATE column if exists (some sheets include)
        df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")
        # if any valid parsed date found, use it
        if df["DATE_PARSED"].notna().any():
            df["DATE"] = df["DATE_PARSED"]
    # If no DATE or parsing failed, build from YEAR/MONTH/DAY
    if "DATE" not in df.columns or df["DATE"].isna().all():
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")

    # Fill small number of missing dates by forward fill/backfill - safe fallback
    if df["DATE"].isna().any():
        df["DATE"] = df["DATE"].fillna(method="ffill").fillna(method="bfill")

    # Revenue column
    df["REVENUE_VAL"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    if df["REVENUE_VAL"].isna().all():
        raise ValueError("Revenue column contains no numeric values.")

    # Remove rows without a valid date or revenue
    df = df.dropna(subset=["DATE", "REVENUE_VAL"]).copy()
    df = df.sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data rows found after cleaning.")

    # Feature engineering for modelling
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df["DOW"] = df["DATE"].dt.dayofweek  # Monday=0 ... Sunday=6
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)
    # Additional features could be added (holidays, payday flag) if available

    # --- 1) Exponential Smoothing forecast (baseline) ---
    try:
        es_model = ExponentialSmoothing(df["REVENUE_VAL"], trend="add", seasonal=None)
        es_fit = es_model.fit(optimized=True)
        es_forecast = es_fit.forecast(30)
        es_fitted = es_fit.fittedvalues
    except Exception:
        # safe fallback to simple mean forecast
        es_forecast = pd.Series(np.repeat(df["REVENUE_VAL"].mean(), 30))
        es_fitted = pd.Series(np.repeat(df["REVENUE_VAL"].mean(), len(df)))

    # --- 2) Prophet forecast ---
    try:
        prophet_df = df[["DATE", "REVENUE_VAL"]].rename(columns={"DATE": "ds", "REVENUE_VAL": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_df)
        future_prophet = prophet_model.make_future_dataframe(periods=30)
        prophet_pred = prophet_model.predict(future_prophet)
        prophet_forecast = prophet_pred.tail(30)["yhat"].values
    except Exception:
        prophet_forecast = np.repeat(df["REVENUE_VAL"].mean(), 30)

    # --- 3) LightGBM on date features (learn seasonality/structure) ---
    try:
        lgb_features = ["YEAR", "MONTH", "DAY", "DOW"]
        X = df[lgb_features]
        y = df["REVENUE_VAL"]
        lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=0.05)
        lgb_model.fit(X, y)
        # Prepare future features
        start_date = pd.Timestamp.today().normalize()
        future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, 31)]
        future_df = pd.DataFrame({"DATE": future_dates})
        future_df["YEAR"] = future_df["DATE"].dt.year
        future_df["MONTH"] = future_df["DATE"].dt.month
        future_df["DAY"] = future_df["DATE"].dt.day
        future_df["DOW"] = future_df["DATE"].dt.dayofweek
        lgb_forecast = lgb_model.predict(future_df[["YEAR", "MONTH", "DAY", "DOW"]])
    except Exception:
        # fallback
        future_dates = [pd.Timestamp.today().normalize() + pd.Timedelta(days=i) for i in range(1, 31)]
        lgb_forecast = np.repeat(df["REVENUE_VAL"].mean(), 30)

    # --- 4) Combine (simple weighted average) ---
    # Using weights that can be tuned; chosen to be robust defaults
    # If any component has NaNs, replace with mean
    def fix_array(arr):
        arr = np.array(arr, dtype=float)
        if np.isnan(arr).any():
            arr = np.where(np.isfinite(arr), arr, df["REVENUE_VAL"].mean())
        return arr

    exp_arr = fix_array(es_forecast)
    prop_arr = fix_array(prophet_forecast)
    lgb_arr = fix_array(lgb_forecast)

    combined = 0.30 * exp_arr + 0.30 * prop_arr + 0.40 * lgb_arr

    # --- 5) Residual correction LightGBM (optional, lightweight) ---
    # We'll train a small residual model if there is enough history
    try:
        # Build residuals on historical data using the same hybrid logic for historical in-sample
        # Create historical hybrid forecast in-sample using a rolling approach would be ideal,
        # but to keep simple and robust for production: use fitted ES + prophet in-sample if available
        in_sample_hybrid = None
        # try to compute prophet in-sample predictions (if prophet fit earlier)
        try:
            pred_full = prophet_model.predict(prophet_df)
            prop_insample = pred_full["yhat"].values
        except Exception:
            prop_insample = np.repeat(df["REVENUE_VAL"].mean(), len(df))

        try:
            es_insample = es_fitted.values if hasattr(es_fitted, "values") else np.repeat(df["REVENUE_VAL"].mean(), len(df))
        except Exception:
            es_insample = np.repeat(df["REVENUE_VAL"].mean(), len(df))

        # Build a simple in-sample hybrid
        in_sample_hybrid = 0.3 * np.array(es_insample) + 0.3 * np.array(prop_insample)
        # residual
        residuals = df["REVENUE_VAL"].values - in_sample_hybrid

        # Features for residual model
        df_resid = df.copy()
        df_resid["resid"] = residuals
        resid_features = ["DOW", "MONTH", "DAY", "IS_WEEKEND"]
        # if not enough unique data, skip
        if df_resid.shape[0] >= 60:
            resid_X = df_resid[resid_features]
            resid_y = df_resid["resid"]
            resid_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
            resid_model.fit(resid_X, resid_y)
            # predict residuals for future_df
            resid_pred_future = resid_model.predict(future_df.assign(IS_WEEKEND=future_df["DOW"].isin([5,6]).astype(int))[resid_features])
            # correct combined forecast
            combined = combined + resid_pred_future
        else:
            # skip residual correction for small datasets
            pass
    except Exception:
        # If anything fails, continue with combined as-is
        pass

    # --- 6) Sunday closed rule (your dataset uses Sundays = 0) ---
    # pandas dayofweek: Sunday = 6
    combined_corrected = []
    for d, val in zip(future_dates, combined):
        dow = d.dayofweek
        if dow == 6:
            combined_corrected.append(0.0)
        else:
            combined_corrected.append(float(safe_float(val, 0.0)))

    # --- 7) Build response data structure (chart_data & daily_forecast) ---
    # Historical chart_data: use the cleaned history
    chart_rows = []
    for _, r in df.iterrows():
        chart_rows.append({
            "date": str(r["DATE"].date()),
            "revenue": safe_float(r["REVENUE_VAL"], 0.0),
            "type": "historical"
        })

    # Future chart points
    for d, v in zip(future_dates, combined_corrected):
        chart_rows.append({
            "date": str(pd.Timestamp(d).date()),
            "revenue": safe_float(v, 0.0),
            "type": "forecast"
        })

    # daily_forecast mapping (date string -> numeric)
    daily_forecast = { str(pd.Timestamp(d).date()): safe_float(v, 0.0) for d, v in zip(future_dates, combined_corrected) }

    # --- 8) Metrics: MAE (compute on last N days in-sample vs ES fitted or other baseline)
    # We'll compute MAE between the last up-to-30 historical days and ES fitted if available,
    # then multiply by 30 to present as monthly MAE (your request).
    mae_val = None
    try:
        # choose in-sample predicted series to compare with actual last 30 days (prefer ES fitted if present)
        insample_pred_source = None
        if 'es_fitted' in locals() and hasattr(es_fitted, "__len__"):
            insample_pred_source = np.array(es_fitted[-30:]) if len(es_fitted) >= 1 else None
        if insample_pred_source is None or len(insample_pred_source) < 1:
            insample_pred_source = np.repeat(df["REVENUE_VAL"].mean(), min(30, len(df)))

        actual_last = np.array(df["REVENUE_VAL"].values[-30:]) if len(df) >= 1 else np.array([])
        # If length mismatch, trim/pad
        n = min(len(actual_last), len(insample_pred_source))
        if n > 0:
            mae_sample = mean_absolute_error(actual_last[-n:], insample_pred_source[-n:])
            mae_val = float(mae_sample)
            mae_monthly = float(round(mae_val * 30.0, 2))  # per-month MAE
        else:
            mae_val = None
            mae_monthly = None
    except Exception:
        mae_val = None
        mae_monthly = None

    total_forecast = float(round(sum(combined_corrected), 2))

    result = {
        "chart_data": chart_rows,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_val, 2) if mae_val is not None else None,
        "mae_monthly": mae_monthly
    }

    # Ensure everything is JSON-friendly (no np.nan, np.float64 etc.)
    # Convert any possible numpy types to python built-ins
    # (We've already sanitized most values, but we do a final pass)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isfinite(v):
                return v
            return 0.0
        if isinstance(obj, float):
            if math.isfinite(obj):
                return obj
            return 0.0
        return obj

    return sanitize(result)


# --------------------------
# Validation route (optional)
# --------------------------
@app.route("/api/revenue/validate", methods=["POST"])
def validate_forecast():
    try:
        payload = request.get_json(force=True)
        actual = payload.get("actual", [])
        predicted = payload.get("predicted", [])
        if not isinstance(actual, list) or not isinstance(predicted, list) or len(actual) == 0 or len(predicted) == 0:
            return jsonify({"status": "error", "message": "Please provide arrays 'actual' and 'predicted'"}), 400
        # align lengths
        n = min(len(actual), len(predicted))
        actual_arr = np.array(actual[:n], dtype=float)
        pred_arr = np.array(predicted[:n], dtype=float)
        mae = float(mean_absolute_error(actual_arr, pred_arr))
        return jsonify({"status": "success", "mae": round(mae, 4), "mae_monthly": round(mae * 30.0, 4)}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


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
        # Return error message but keep JSON safe
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/history", methods=["GET", "OPTIONS"])
def get_history():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        # currently no saved history storage; return empty list to frontend
        return jsonify({"status": "success", "data": []}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        df_out = pd.DataFrame(list(result["daily_forecast"].items()), columns=["Date", "Forecasted_Revenue"])
        # Add summary row
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
