# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import io
import traceback
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Path to Excel file (adjust if different)
EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# ---------------------------
# Helpers
# ---------------------------
def _find_date_and_revenue_columns(df):
    """Return (date_col, revenue_col) in uppercase column names or raise ValueError."""
    cols = [c.strip().upper() for c in df.columns]

    # If YEAR/MONTH/DAY present
    if {"YEAR", "MONTH", "DAY"}.issubset(cols):
        revenue_col = None
        if "REVENUE" in cols:
            revenue_col = next(c for c in df.columns if c.strip().upper() == "REVENUE")
        elif "AMOUNT" in cols:
            revenue_col = next(c for c in df.columns if c.strip().upper() == "AMOUNT")
        else:
            raise ValueError("When using YEAR/MONTH/DAY the file must also include a revenue column named REVENUE or AMOUNT.")
        return ("YMD", revenue_col)

    # Otherwise try to find a date-like column and revenue
    # Date candidates: any column name that contains 'DATE' or 'DS'
    date_candidates = [c for c in df.columns if "DATE" in c.strip().upper() or c.strip().upper() == "DS"]
    revenue_candidates = [c for c in df.columns if c.strip().upper() in ("REVENUE", "AMOUNT")]

    if date_candidates and revenue_candidates:
        return (date_candidates[0], revenue_candidates[0])

    raise ValueError("Excel must have YEAR/MONTH/DAY + REVENUE/AMOUNT OR a DATE column + REVENUE/AMOUNT.")


def load_standard_df(path=EXCEL_PATH):
    """Load excel and return DataFrame with columns ['ds', 'y'] (ds datetime, y float)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    # read first sheet; support xlsx
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl" if str(path).lower().endswith("xlsx") else None)

    # Normalize column names for matching but keep original for extraction
    try:
        kind, col = _find_date_and_revenue_columns(df)
    except ValueError as e:
        raise

    if kind == "YMD":
        # use YEAR/MONTH/DAY columns (case-insensitive)
        # find original column names
        col_map = {c.strip().upper(): c for c in df.columns}
        ycol = col_map["REVENUE"] if "REVENUE" in col_map else col_map.get("AMOUNT")
        ycol = ycol  # original-case name
        year_col = col_map["YEAR"]; month_col = col_map["MONTH"]; day_col = col_map["DAY"]

        # coerce numeric
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df[day_col] = pd.to_numeric(df[day_col], errors="coerce")
        df["ds"] = pd.to_datetime(df[[year_col, month_col, day_col]], errors="coerce")
        df["y"] = pd.to_numeric(df[ycol], errors="coerce")
        df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows after parsing YEAR/MONTH/DAY + revenue.")
        return df[["ds", "y"]]

    # date column path
    date_col = col
    revenue_col = _find_date_and_revenue_columns(df)[1] if isinstance(_find_date_and_revenue_columns(df), tuple) else None
    # parse
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")
    df = df.dropna(subset=[date_col, revenue_col]).sort_values(date_col).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after parsing DATE + revenue.")
    df = df.rename(columns={date_col: "ds", revenue_col: "y"})
    return df[["ds", "y"]]


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------
# Forecast + metrics function
# ---------------------------
def generate_forecast_and_metrics(forecast_days=30):
    """
    Returns a dict: {
      daily_forecast: {date_str: value,...},
      total_forecast: float,
      metrics: {"MAE":float,"RMSE":float,"MAPE":float-or-null}
    }
    """
    df = load_standard_df(EXCEL_PATH)  # may raise

    # keep minimum size check
    if len(df) < 8:
        raise ValueError("Not enough historical rows (need at least ~8).")

    # Backtest split: last N days for validation
    backtest_size = min(30, max(1, int(len(df) * 0.2)))
    train_df = df.iloc[:-backtest_size].copy()
    test_df = df.iloc[-backtest_size:].copy()

    # Exponential Smoothing on train
    try:
        es_model = ExponentialSmoothing(train_df["y"], trend="add", seasonal=None).fit()
        es_test_pred = es_model.forecast(backtest_size)
    except Exception:
        es_test_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # Prophet on train
    try:
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(train_df.rename(columns={"ds": "ds", "y": "y"}))
        future = prophet_model.make_future_dataframe(periods=backtest_size)
        prophet_test_pred = prophet_model.predict(future).tail(backtest_size)["yhat"].values
    except Exception:
        prophet_test_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # LightGBM on train
    try:
        def mk_feats(dframe):
            return pd.DataFrame({
                "dayofweek": dframe["ds"].dt.dayofweek,
                "month": dframe["ds"].dt.month,
                "year": dframe["ds"].dt.year
            })
        X_train = mk_feats(train_df)
        y_train = train_df["y"].values
        X_test = mk_feats(test_df)
        lgbm = lgb.LGBMRegressor(objective="regression", n_estimators=100)
        lgbm.fit(X_train, y_train)
        lgb_test_pred = lgbm.predict(X_test)
    except Exception:
        lgb_test_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # Hybrid validation prediction
    hybrid_test = (np.array(es_test_pred) + np.array(prophet_test_pred) + np.array(lgb_test_pred)) / 3.0

    # Metrics on test period
    actual = test_df["y"].values
    mae = float(mean_absolute_error(actual, hybrid_test))
    rmse = float(mean_squared_error(actual, hybrid_test) ** 0.5)
    mape_val = safe_mape(actual, hybrid_test)
    mape = float(round(mape_val, 4)) if mape_val is not None else None

    # -------------------------
    # Forecast forward using full data
    # -------------------------
    # ES full
    try:
        es_full = ExponentialSmoothing(df["y"], trend="add", seasonal=None).fit()
        es_future = es_full.forecast(forecast_days)
    except Exception:
        es_future = np.repeat(df["y"].mean(), forecast_days)

    # Prophet full
    try:
        prophet_full = Prophet(daily_seasonality=True)
        prophet_full.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        future_full = prophet_full.make_future_dataframe(periods=forecast_days)
        prophet_future = prophet_full.predict(future_full).tail(forecast_days)["yhat"].values
    except Exception:
        prophet_future = np.repeat(df["y"].mean(), forecast_days)

    # LightGBM full
    try:
        features_full = pd.DataFrame({
            "dayofweek": df["ds"].dt.dayofweek,
            "month": df["ds"].dt.month,
            "year": df["ds"].dt.year
        })
        lgb_full = lgb.LGBMRegressor(objective="regression", n_estimators=100)
        lgb_full.fit(features_full, df["y"].values)

        last_date = df["ds"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_feats = pd.DataFrame({
            "dayofweek": [d.weekday() for d in future_dates],
            "month": [d.month for d in future_dates],
            "year": [d.year for d in future_dates]
        })
        lgb_future = lgb_full.predict(future_feats)
    except Exception:
        last_date = df["ds"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        lgb_future = np.repeat(df["y"].mean(), forecast_days)

    # combine forecasts (average)
    combined = (np.array(es_future) + np.array(prophet_future) + np.array(lgb_future)) / 3.0

    # Set Sundays to 0 (weekday()==6)
    for i, d in enumerate(future_dates):
        if d.weekday() == 6:
            combined[i] = 0.0

    # Build daily_forecast dict & totals
    daily_forecast = {d.strftime("%Y-%m-%d"): float(round(float(val), 2)) for d, val in zip(future_dates, combined)}
    total_forecast = float(round(float(np.sum(combined)), 2))

    return {
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "metrics": {
            "MAE": float(round(mae, 2)),
            "RMSE": float(round(rmse, 2)),
            "MAPE": float(round(mape, 2)) if mape is not None else None
        }
    }


# ---------------------------
# API Routes
# ---------------------------
@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def api_forecast():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        res = generate_forecast_and_metrics(forecast_days=30)
        return jsonify({"status": "success", "data": res}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/history", methods=["GET", "OPTIONS"])
def api_history():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        # If you have a stored history file, return it; otherwise return empty list
        history_file = "Revenue_Forecast_History.xlsx"
        if os.path.exists(history_file):
            df = pd.read_excel(history_file)
            data = df.to_dict(orient="records")
            return jsonify({"status": "success", "data": data}), 200
        else:
            return jsonify({"status": "success", "data": []}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def api_download():
    try:
        # produce a temporary excel of latest forecast
        res = generate_forecast_and_metrics(forecast_days=30)
        daily = res["daily_forecast"]
        df_out = pd.DataFrame(list(daily.items()), columns=["Date", "Forecasted_Revenue"])
        # add totals row
        df_out.loc[len(df_out)] = ["Total (₱)", res["total_forecast"]]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Forecast_30_Days")
            metrics_df = pd.DataFrame(list(res["metrics"].items()), columns=["Metric", "Value"])
            metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        output.seek(0)

        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API active"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
