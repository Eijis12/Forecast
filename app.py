from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import datetime
import traceback
import os

# forecasting libs
from prophet import Prophet
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Path to your Excel file on the server
EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# -------------------------
# Helper: robustly load the Excel into a standard dataframe with columns: ds (datetime), y (revenue)
# Accepts:
#  - Columns YEAR, MONTH, DAY + REVENUE or AMOUNT
#  - Or single DATE column and REVENUE/AMOUNT
# -------------------------
def load_revenue_dataframe(path=EXCEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at path: {path}")

    # read
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl" if path.lower().endswith(("xlsx", "xlsm")) else None)

    # normalize column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    cols = set(df.columns)

    # prioritize explicit date columns YEAR/MONTH/DAY
    if {"YEAR", "MONTH", "DAY"}.issubset(cols):
        # revenue can be called REVENUE or AMOUNT; accept either
        if "REVENUE" in cols:
            revenue_col = "REVENUE"
        elif "AMOUNT" in cols:
            revenue_col = "AMOUNT"
        else:
            raise ValueError("Excel must include a revenue column named 'REVENUE' or 'AMOUNT' when using YEAR/MONTH/DAY.")

        # make numeric and safe parse
        for c in ("YEAR", "MONTH", "DAY"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["ds"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
        df["y"] = pd.to_numeric(df[revenue_col], errors="coerce")
        df = df.dropna(subset=["ds"])  # remove rows we couldn't parse a date for
        df = df.dropna(subset=["y"]).sort_values("ds")
        if df.empty:
            raise ValueError("After parsing YEAR/MONTH/DAY and the revenue column, no valid rows remain.")
        return df[["ds", "y"]]

    # fallback: single DATE column
    # possible names: DATE, DS, DAY_DATE etc — find something that parses as a date
    date_candidates = [c for c in df.columns if "DATE" in c or c in ("DS", "D")]
    if date_candidates:
        date_col = date_candidates[0]
        # revenue column:
        if "REVENUE" in cols:
            revenue_col = "REVENUE"
        elif "AMOUNT" in cols:
            revenue_col = "AMOUNT"
        else:
            raise ValueError("Excel must include a revenue column named 'REVENUE' or 'AMOUNT' when using a DATE column.")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")
        df = df.dropna(subset=[date_col]).dropna(subset=[revenue_col]).sort_values(date_col)
        if df.empty:
            raise ValueError("After parsing DATE and revenue columns, no valid rows remain.")
        df = df.rename(columns={date_col: "ds", revenue_col: "y"})
        return df[["ds", "y"]]

    # If reached here, we couldn't find date-like columns
    raise ValueError("Excel must have columns: YEAR, MONTH, DAY, and REVENUE/AMOUNT OR a DATE column plus REVENUE/AMOUNT.")


# -------------------------
# Metrics helpers (safe)
# -------------------------
def safe_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# -------------------------
# Forecasting: hybrid & metrics
# -------------------------
def generate_forecast_and_metrics(forecast_days=30):
    # load and standardize data
    df = load_revenue_dataframe(EXCEL_PATH)  # returns columns ds (datetime), y (float)
    df = df.reset_index(drop=True)

    # if dataset is very short, raise helpful error
    if len(df) < 10:
        raise ValueError("Not enough historical data (need at least ~10 rows).")

    # --- backtest: hold out last N days for validation (use min(30, 20% of data) )
    backtest_size = min(30, max(1, int(len(df) * 0.2)))
    train_df = df.iloc[:-backtest_size].copy()
    test_df = df.iloc[-backtest_size:].copy()

    # ---------- Exponential Smoothing (on train)
    try:
        es = ExponentialSmoothing(train_df["y"], trend="add", seasonal=None).fit()
        es_test_pred = es.forecast(backtest_size)
    except Exception:
        # fallback to train mean
        es_test_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # ---------- Prophet (on train)
    try:
        prophet_train = train_df.rename(columns={"ds": "ds", "y": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=backtest_size)
        prophet_pred = prophet_model.predict(future).tail(backtest_size)["yhat"].values
    except Exception:
        prophet_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # ---------- LightGBM (on train)
    try:
        # prepare lag features simple: dayofweek, month, year
        def mk_features(df_):
            return pd.DataFrame({
                "dayofweek": df_["ds"].dt.dayofweek,
                "month": df_["ds"].dt.month,
                "year": df_["ds"].dt.year
            })
        X_train = mk_features(train_df)
        y_train = train_df["y"].values
        X_test = mk_features(test_df)
        lgbm = lgb.LGBMRegressor(objective="regression", n_estimators=100)
        lgbm.fit(X_train, y_train)
        lgb_test_pred = lgbm.predict(X_test)
    except Exception:
        lgb_test_pred = np.repeat(train_df["y"].mean(), backtest_size)

    # --- Hybrid (validation predictions)
    hybrid_test_pred = (np.array(es_test_pred) + np.array(prophet_pred) + np.array(lgb_test_pred)) / 3.0

    # --- Metrics (on test_df)
    actual = test_df["y"].values
    mae = float(mean_absolute_error(actual, hybrid_test_pred))
    rmse = float(mean_squared_error(actual, hybrid_test_pred, squared=False))
    mape_val = safe_mape(actual, hybrid_test_pred)
    mape = float(round(mape_val, 4)) if mape_val is not None else None

    # ---------- Now produce final future forecast for next forecast_days
    # We'll train models on the full dataset and forecast forward
    # Exponential Smoothing full
    try:
        es_full = ExponentialSmoothing(df["y"], trend="add", seasonal=None).fit()
        es_future = es_full.forecast(forecast_days)
    except Exception:
        es_future = np.repeat(df["y"].mean(), forecast_days)

    # Prophet full
    try:
        prophet_full = Prophet(daily_seasonality=True)
        prophet_full.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        future_df_prophet = prophet_full.make_future_dataframe(periods=forecast_days)
        prophet_future = prophet_full.predict(future_df_prophet).tail(forecast_days)["yhat"].values
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
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_features = pd.DataFrame({
            "dayofweek": [d.weekday() for d in future_dates],
            "month": [d.month for d in future_dates],
            "year": [d.year for d in future_dates]
        })
        lgb_future = lgb_full.predict(future_features)
    except Exception:
        # fallback
        future_dates = [df["ds"].max() + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
        lgb_future = np.repeat(df["y"].mean(), forecast_days)

    # Combine three forecasts (simple average)
    combined = (np.array(es_future) + np.array(prophet_future) + np.array(lgb_future)) / 3.0

    # Set Sundays to 0 (Sunday weekday() == 6)
    for i, d in enumerate(future_dates):
        if d.weekday() == 6:
            combined[i] = 0.0

    # Build daily_forecast dict
    daily_forecast = {d.strftime("%Y-%m-%d"): float(round(float(val), 2)) for d, val in zip(future_dates, combined)}
    total_forecast = float(round(float(np.sum(combined)), 2))

    metrics = {
        "MAE": float(round(mae, 2)),
        "RMSE": float(round(rmse, 2)),
        # ensure MAPE is JSON-friendly; if None set to None
        "MAPE": float(round(mape, 2)) if mape is not None else None
    }

    return {
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "metrics": metrics
    }


# -------------------------
# API Routes
# -------------------------
@app.route("/api/revenue/forecast", methods=["POST", "OPTIONS"])
def api_forecast():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        result = generate_forecast_and_metrics()
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        # log server side
        traceback.print_exc()
        # return readable error to frontend
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def api_download():
    try:
        res = generate_forecast_and_metrics()
        daily = res["daily_forecast"]
        df_out = pd.DataFrame(list(daily.items()), columns=["Date", "Forecasted_Revenue"])
        # add summary
        df_out.loc[len(df_out)] = ["Total (₱)", res["total_forecast"]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Forecast")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="RevenueForecast.xlsx")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API (robust) active"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
