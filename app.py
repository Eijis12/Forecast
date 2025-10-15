# ============================================================
# app.py — Recursive Hybrid Forecast Model (Prophet + ES + LGBM)
# ============================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io, traceback, os, math
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://campbelldentalsystem.site", "*"]}})

EXCEL_PATH = "Dental_Revenue_2425.xlsx"


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
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


def ensure_list_of_floats(arr):
    return [safe_float(v, 0.0) for v in list(arr)]


# ------------------------------------------------------------
# Recursive Forecast Generator
# ------------------------------------------------------------
def generate_forecast():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    if "REVENUE" in df.columns:
        df = df.rename(columns={"REVENUE": "AMOUNT"})

    required = {"YEAR", "MONTH", "DAY", "AMOUNT"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required}")

    # Clean numeric
    for col in ["YEAR", "MONTH", "DAY", "AMOUNT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Construct DATE
    if "DATE" in df.columns:
        df["DATE_PARSED"] = pd.to_datetime(df["DATE"], errors="coerce")
        if df["DATE_PARSED"].notna().any():
            df["DATE"] = df["DATE_PARSED"]
    if "DATE" not in df.columns or df["DATE"].isna().all():
        df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    df = df.dropna(subset=["DATE", "AMOUNT"]).copy()
    df["REVENUE_VAL"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    df = df.dropna(subset=["REVENUE_VAL"]).sort_values("DATE").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows after cleaning.")

    # Feature engineering
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df["DOW"] = df["DATE"].dt.dayofweek
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)

    # -------------------
    # 1️⃣ Base models fit
    # -------------------
    # Exponential Smoothing
    try:
        es_model = ExponentialSmoothing(df["REVENUE_VAL"], trend="add", seasonal=None)
        es_fit = es_model.fit(optimized=True)
        es_fitted = es_fit.fittedvalues
    except Exception:
        es_fit = None
        es_fitted = pd.Series(np.repeat(df["REVENUE_VAL"].mean(), len(df)))

    # Prophet
    try:
        p_df = df[["DATE", "REVENUE_VAL"]].rename(columns={"DATE": "ds", "REVENUE_VAL": "y"})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(p_df)
    except Exception:
        prophet_model = None

    # LightGBM (trained on features)
    lgb_features = ["YEAR", "MONTH", "DAY", "DOW", "IS_WEEKEND"]
    X = df[lgb_features]
    y = df["REVENUE_VAL"]
    lgb_model = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05)
    lgb_model.fit(X, y)

    # ------------------------------------------------------------
    # 2️⃣ Recursive Forecast Loop — day by day for next 30 days
    # ------------------------------------------------------------
    start_date = df["DATE"].max()
    future_rows = []
    last_revenue = df["REVENUE_VAL"].iloc[-1]

    for i in range(1, 31):
        next_date = start_date + pd.Timedelta(days=i)
        dow = next_date.dayofweek
        is_weekend = 1 if dow in [5, 6] else 0

        # skip Sundays (closed)
        if dow == 6:
            pred_value = 0.0
        else:
            # Get base predictions from models
            es_pred = es_fit.forecast(1)[0] if es_fit is not None else df["REVENUE_VAL"].mean()
            prop_pred = (
                prophet_model.predict(pd.DataFrame({"ds": [next_date]}))["yhat"].iloc[0]
                if prophet_model else df["REVENUE_VAL"].mean()
            )
            lgb_pred = lgb_model.predict(
                pd.DataFrame(
                    {"YEAR": [next_date.year], "MONTH": [next_date.month],
                     "DAY": [next_date.day], "DOW": [dow], "IS_WEEKEND": [is_weekend]}
                )
            )[0]

            # Weighted hybrid (tunable weights)
            pred_value = 0.3 * es_pred + 0.3 * prop_pred + 0.4 * lgb_pred

            # Residual correction (based on last 30 errors)
            if len(df) > 30:
                residuals = df["REVENUE_VAL"].values[-30:] - es_fitted.values[-30:]
                resid_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05)
                resid_model.fit(X.tail(30), residuals)
                resid_future = resid_model.predict(
                    pd.DataFrame(
                        {"YEAR": [next_date.year], "MONTH": [next_date.month],
                         "DAY": [next_date.day], "DOW": [dow], "IS_WEEKEND": [is_weekend]}
                    )
                )[0]
                pred_value += resid_future

        # append to results
        future_rows.append({"DATE": next_date, "FORECASTED_REVENUE": safe_float(pred_value, 0.0)})
        # recursively append to df for next iteration
        new_row = {
            "DATE": next_date,
            "REVENUE_VAL": safe_float(pred_value, 0.0),
            "YEAR": next_date.year,
            "MONTH": next_date.month,
            "DAY": next_date.day,
            "DOW": dow,
            "IS_WEEKEND": is_weekend,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # ------------------------------------------------------------
    # 3️⃣ Metrics — MAE per day ×30 for monthly view
    # ------------------------------------------------------------
    try:
        if len(es_fitted) >= 30:
            mae_day = mean_absolute_error(df["REVENUE_VAL"].iloc[-60:-30], es_fitted.iloc[-30:])
            mae_month = round(mae_day * 30.0, 2)
        else:
            mae_day, mae_month = None, None
    except Exception:
        mae_day, mae_month = None, None

    # ------------------------------------------------------------
    # 4️⃣ Prepare JSON-safe response
    # ------------------------------------------------------------
    daily_forecast = {
        str(r["DATE"].date()): safe_float(r["FORECASTED_REVENUE"], 0.0)
        for _, r in pd.DataFrame(future_rows).iterrows()
    }
    total_forecast = round(sum(daily_forecast.values()), 2)

    chart_rows = [
        {"date": str(d.date()), "revenue": safe_float(v, 0.0), "type": "historical"}
        for d, v in zip(df["DATE"].iloc[:-30], df["REVENUE_VAL"].iloc[:-30])
    ]
    for d, v in daily_forecast.items():
        chart_rows.append({"date": d, "revenue": v, "type": "forecast"})

    result = {
        "chart_data": chart_rows,
        "daily_forecast": daily_forecast,
        "total_forecast": total_forecast,
        "mae_daily": round(mae_day, 2) if mae_day is not None else None,
        "mae_monthly": mae_month,
    }

    return result


# ------------------------------------------------------------
# Flask API endpoints
# ------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "Recursive Hybrid Forecast API active"}), 200


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


@app.route("/api/revenue/history", methods=["GET"])
def get_history():
    return jsonify({"status": "success", "data": []}), 200


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
