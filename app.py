from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
import os
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

# Dataset path (same name you provided)
EXCEL_PATH = "Dental_Revenue_2425.xlsx"

# ------------------------------
# Helpers
# ------------------------------
def safe_read_excel(path):
    """Read excel and normalize column names (uppercase, strip)."""
    df = pd.read_excel(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def ensure_cols(df, required):
    cols = set(df.columns)
    missing = required - cols
    if missing:
        raise ValueError(f"Excel file must have columns: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}")

def make_features_for_df(df):
    """Add basic features used by the residual model and recursive forecasting."""
    df = df.copy()
    df['day_of_week'] = df['DATE'].dt.weekday  # Monday=0 .. Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    # Simple payday heuristic (1st and 15th as paydays) — tweak if needed
    df['is_payday'] = df['DATE'].dt.day.isin([1, 15]).astype(int)
    df['month'] = df['DATE'].dt.month
    # Placeholder: no holiday list provided; set 0
    df['is_holiday'] = 0
    return df

# ------------------------------
# Core forecasting pipeline
# ------------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # Read and normalize
    df = safe_read_excel(EXCEL_PATH)

    # Accept REVENUE column naming consistency (you said columns are DATE | YEAR | MONTH | DAY | REVENUE)
    # Some sheets might call it AMOUNT -> map it, but prefer REVENUE as you gave.
    if 'REVENUE' not in df.columns and 'AMOUNT' in df.columns:
        df.rename(columns={'AMOUNT': 'REVENUE'}, inplace=True)

    # Required columns for this pipeline
    required = {"DATE", "YEAR", "MONTH", "DAY", "REVENUE"}
    ensure_cols(df, required)

    # Convert types
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
    df['DAY'] = pd.to_numeric(df['DAY'], errors='coerce')
    df['REVENUE'] = pd.to_numeric(df['REVENUE'], errors='coerce')

    # Drop rows missing date or revenue
    df = df.dropna(subset=['DATE', 'REVENUE']).sort_values('DATE').reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data rows found. Please check your Excel file values.")

    # Keep a copy for historical chart data
    hist_df = df[['DATE', 'REVENUE']].copy()
    hist_df.rename(columns={'DATE': 'Date', 'REVENUE': 'Revenue'}, inplace=True)
    hist_df['Type'] = 'historical'

    # --- 1) Exponential Smoothing (ES) baseline ---
    es_forecast_future = None
    es_fitted = None
    try:
        # Use weekly seasonality when possible (period=7) — add seasonal if data long enough
        seasonal = 7 if len(df) >= 14 else None
        es_model = ExponentialSmoothing(df['REVENUE'], trend='add', seasonal='add' if seasonal else None, seasonal_periods=seasonal)
        es_fit = es_model.fit(optimized=True, use_boxcox=False, remove_bias=False)
        es_fitted = es_fit.fittedvalues
    except Exception:
        # fallback: use rolling mean as fitted
        es_fit = None
        es_fitted = pd.Series(df['REVENUE'].rolling(window=min(7, max(1, len(df))).min(), min_periods=1).mean().values, index=df.index)

    # --- 2) Prophet model to capture seasonality/trends ---
    prophet_preds_hist = None
    try:
        prophet_df = pd.DataFrame({'ds': df['DATE'], 'y': df['REVENUE']})
        prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        # Fit quietly
        prophet_model.fit(prophet_df)
        prophet_hist = prophet_model.predict(prophet_df[['ds']])
        prophet_preds_hist = prophet_hist['yhat'].values
    except Exception:
        prophet_model = None
        prophet_preds_hist = np.array(df['REVENUE'].values)  # fallback: identity

    # --- 3) Construct hybrid forecast on historical (before correction) ---
    # Use simple averaging weights between ES and Prophet for the hybrid baseline
    # If es_fitted or prophet_preds_hist aren't same length, align via index
    try:
        es_vals = np.array(es_fitted)
    except Exception:
        es_vals = np.array(df['REVENUE'].values)

    if len(es_vals) != len(prophet_preds_hist):
        # align lengths (should normally match)
        minlen = min(len(es_vals), len(prophet_preds_hist))
        es_vals = es_vals[-minlen:]
        prophet_preds_hist = prophet_preds_hist[-minlen:]
        df = df.iloc[-minlen:].copy()
        hist_df = hist_df.iloc[-minlen:].copy()

    hybrid_hist = 0.5 * es_vals + 0.5 * prophet_preds_hist

    # --- 4) Residual LightGBM correction trained on hybrid residuals ---
    # Prepare features
    features_df = make_features_for_df(df)
    X = features_df[['day_of_week', 'is_weekend', 'is_payday', 'month', 'is_holiday']].astype(float)
    residuals = df['REVENUE'].values - hybrid_hist

    lgb_model = None
    try:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        lgb_model.fit(X, residuals)
        resid_pred_hist = lgb_model.predict(X)
        hybrid_corrected_hist = hybrid_hist + resid_pred_hist
    except Exception:
        # fallback: no correction
        lgb_model = None
        resid_pred_hist = np.zeros_like(hybrid_hist)
        hybrid_corrected_hist = hybrid_hist

    # Compute MAE on last N historical points (choose last 30 or available)
    n_eval = min(30, len(df))
    try:
        mae_val = mean_absolute_error(df['REVENUE'].values[-n_eval:], hybrid_corrected_hist[-n_eval:])
        mae_val = float(np.round(mae_val, 2))
    except Exception:
        mae_val = None

    # --- 5) Recursive future forecasting for next 30 days ---
    horizon = 30
    start_date = pd.Timestamp.today().normalize()
    future_dates = [start_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]

    # ES future forecast (if es_fit available)
    try:
        if es_fit is not None:
            es_fore = es_fit.forecast(horizon)
            es_future = np.array(es_fore)
        else:
            es_future = np.repeat(df['REVENUE'].mean(), horizon)
    except Exception:
        es_future = np.repeat(df['REVENUE'].mean(), horizon)

    # Prophet future forecast
    try:
        if prophet_model is not None:
            future_prophet = prophet_model.make_future_dataframe(periods=horizon, freq='D')
            prophet_forecast_df = prophet_model.predict(future_prophet)
            # prophet_forecast_df contains historical+future; extract the tail(horizon) yhat
            prophet_future = prophet_forecast_df['yhat'].tail(horizon).values
        else:
            prophet_future = np.repeat(df['REVENUE'].mean(), horizon)
    except Exception:
        prophet_future = np.repeat(df['REVENUE'].mean(), horizon)

    # Simple hybrid future (un-corrected)
    hybrid_future = 0.5 * es_future + 0.5 * prophet_future

    # We'll now apply residual correction per-day using lgb_model (if available)
    corrected_future = []
    for i, d in enumerate(future_dates):
        dow = d.weekday()
        is_weekend = 1 if dow in (5, 6) else 0
        is_payday = 1 if d.day in (1, 15) else 0
        month = d.month
        is_holiday = 0

        features_row = np.array([[dow, is_weekend, is_payday, month, is_holiday]], dtype=float)

        base_val = hybrid_future[i]

        if lgb_model is not None:
            try:
                resid_pred = float(lgb_model.predict(features_row)[0])
            except Exception:
                resid_pred = 0.0
        else:
            resid_pred = 0.0

        corrected_val = base_val + resid_pred

        # Respect Sundays are zero in your dataset (Sunday = weekday() == 6)
        if d.weekday() == 6:
            corrected_val = 0.0

        # never negative forecast
        if np.isnan(corrected_val) or corrected_val is None:
            corrected_val = float(df['REVENUE'].mean())
        corrected_future.append(float(max(0.0, float(round(corrected_val, 2)))))

    # Build combined chart data (historical + forecast)
    future_df_chart = pd.DataFrame({
        'Date': [d for d in future_dates],
        'Revenue': corrected_future,
        'Type': ['forecast'] * horizon
    })
    combined_chart_df = pd.concat([
        hist_df.rename(columns={'Date': 'Date', 'Revenue': 'Revenue'})[['Date', 'Revenue', 'Type']],
        future_df_chart[['Date', 'Revenue', 'Type']]
    ], ignore_index=True)

    # Format chart_data for JSON output
    chart_data = [
        {"date": str(row['Date'].date()), "revenue": round(float(row['Revenue']), 2), "type": row['Type']}
        for _, row in combined_chart_df.iterrows()
    ]

    # daily_forecast mapping
    daily_forecast = {
        str(d.date()): round(float(v), 2) for d, v in zip(future_dates, corrected_future)
    }

    total_forecast = float(round(sum(corrected_future), 2))

    # Keep an accuracy placeholder (front-end expects json.data.accuracy sometimes)
    accuracy_placeholder = None

    result = {
        "chart_data": chart_data,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae": mae_val,
        "accuracy": accuracy_placeholder
    }

    return result

# ------------------------------
# Validation route (client can send arrays to compute MAE)
# ------------------------------
@app.route("/api/revenue/validate", methods=["POST"])
def validate_forecast():
    try:
        body = request.get_json(force=True)
        actual = body.get('actual')
        predicted = body.get('predicted')
        if actual is None or predicted is None:
            return jsonify({"status": "error", "message": "Provide 'actual' and 'predicted' arrays"}), 400
        actual_arr = np.array(actual, dtype=float)
        predicted_arr = np.array(predicted, dtype=float)
        if actual_arr.shape != predicted_arr.shape:
            return jsonify({"status": "error", "message": "actual and predicted must have same length"}), 400
        mae_val = float(mean_absolute_error(actual_arr, predicted_arr))
        return jsonify({"status": "success", "mae": round(mae_val, 2)}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------
# Main API routes (forecast, history, download)
# ------------------------------
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
        # No historical saved forecasts storage yet — return empty list
        return jsonify({"status": "success", "data": []}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        result = generate_forecast()
        # Build a DataFrame for the 30-day forecast
        df_out = pd.DataFrame(list(result['daily_forecast'].items()), columns=['Date', 'Forecasted_Revenue'])
        # Add summary row
        df_out.loc[len(df_out)] = ['Total (₱)', result['total_forecast']]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Forecast_30_Days')
        output.seek(0)

        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    # Development server for Render or local testing; in production use a WSGI server if desired.
    app.run(host="0.0.0.0", port=5000)
