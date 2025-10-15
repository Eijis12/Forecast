from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import datetime
import io, os, traceback
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# ============================================================
# 📂 CONFIGURATION
# ============================================================
DATA_FILE = "Dental_Revenue_2425.xlsx"
FORECAST_DAYS = 30
HISTORY_FILE = "forecast_history.csv"


# ============================================================
# 🧠 HELPER FUNCTIONS
# ============================================================
def load_data():
    """Load and preprocess revenue dataset"""
    try:
        df = pd.read_excel(DATA_FILE)
    except Exception as e:
        raise Exception(f"Failed to read Excel file: {e}")

    # Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    if not {"year", "month", "day", "amount"}.issubset(df.columns):
        raise Exception("Excel file must have 'Year', 'Month', 'Day', 'Amount' columns")

    df["ds"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values("ds")

    df = df.rename(columns={"amount": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)
    df["y_smooth"] = df["y"].rolling(3, min_periods=1).mean()

    return df


def build_features(df):
    """Add time-based features for regressors"""
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_payday"] = df["ds"].dt.day.isin([15, 30]).astype(int)
    df["month"] = df["ds"].dt.month
    df["is_holiday"] = df["ds"].dt.dayofweek == 6  # placeholder for Sunday=holiday
    return df


def future_dates(last_date, n=FORECAST_DAYS):
    """Generate next n dates"""
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=n)


# ============================================================
# 🔮 FORECAST GENERATION
# ============================================================
def generate_forecast():
    df = load_data()
    df = build_features(df)

    # ========================================================
    # 1️⃣ EXPONENTIAL SMOOTHING MODEL
    # ========================================================
    try:
        es_model = ExponentialSmoothing(df["y_smooth"], trend="add", seasonal=None)
        es_fit = es_model.fit()
        df["ES_forecast"] = es_fit.fittedvalues
    except Exception as e:
        raise Exception(f"ES model failed: {e}")

    # ========================================================
    # 2️⃣ PROPHET MODEL
    # ========================================================
    try:
        prophet_df = df[["ds", "y_smooth"]].rename(columns={"ds": "ds", "y_smooth": "y"})
        prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=FORECAST_DAYS)
        forecast = prophet_model.predict(future)
        prophet_pred = forecast[["ds", "yhat"]].set_index("ds")
        df["Prophet_forecast"] = prophet_pred.loc[df["ds"], "yhat"].values
    except Exception as e:
        raise Exception(f"Prophet model failed: {e}")

    # ========================================================
    # 3️⃣ LIGHTGBM REGRESSOR (Base learner)
    # ========================================================
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_payday"] = df["ds"].dt.day.isin([15, 30]).astype(int)
    df["month"] = df["ds"].dt.month
    df["is_holiday"] = df["ds"].dt.dayofweek == 6

    features = ["day_of_week", "is_weekend", "is_payday", "month", "is_holiday"]

    X = df[features]
    y = df["y_smooth"]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgb_model.fit(X, y)
    df["LGBM_forecast"] = lgb_model.predict(X)

    # ========================================================
    # 4️⃣ HYBRID BASE FORECAST
    # ========================================================
    df["Hybrid_forecast"] = (
        0.3 * df["ES_forecast"] + 0.3 * df["Prophet_forecast"] + 0.4 * df["LGBM_forecast"]
    )

    # ========================================================
    # 5️⃣ RESIDUAL LIGHTGBM CORRECTION
    # ========================================================
    df["Residual"] = df["y_smooth"] - df["Hybrid_forecast"]
    X_train = df[features]
    y_train = df["Residual"]

    lgb_residual = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgb_residual.fit(X_train, y_train)

    df["Residual_pred"] = lgb_residual.predict(X_train)
    df["Hybrid_corrected"] = df["Hybrid_forecast"] + df["Residual_pred"]

    # ========================================================
    # 6️⃣ MODEL EVALUATION
    # ========================================================
    actual = df["y_smooth"].values
    predicted = df["Hybrid_corrected"].values

    mse_corr = mean_squared_error(actual, predicted)
    mae_corr = mean_absolute_error(actual, predicted)
    eps = 1e-3
    mask = np.abs(actual) > eps
    mape_corr = (
        np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        if np.sum(mask) > 0
        else np.nan
    )

    # ========================================================
    # 7️⃣ FUTURE FORECAST (Recursive)
    # ========================================================
    last_date = df["ds"].iloc[-1]
    future_df = pd.DataFrame({"ds": future_dates(last_date)})
    future_df = build_features(future_df)

    # Prophet future
    prophet_future = forecast.set_index("ds").loc[future_df["ds"], "yhat"].values
    # ES future
    es_future = es_fit.forecast(FORECAST_DAYS)
    # LightGBM base
    X_future = future_df[features]
    lgb_future = lgb_model.predict(X_future)

    future_df["Hybrid_future_forecast"] = (
        0.3 * es_future + 0.3 * prophet_future + 0.4 * lgb_future
    )

    # Residual correction for future
    future_df["Residual_pred"] = lgb_residual.predict(X_future)
    future_df["Hybrid_future_corrected"] = (
        future_df["Hybrid_future_forecast"] + future_df["Residual_pred"]
    )

    # Sundays = 0
    future_df["Hybrid_future_corrected"] = np.where(
        future_df["ds"].dt.dayofweek == 6, 0, future_df["Hybrid_future_corrected"]
    )

    # ========================================================
    # 8️⃣ RESULTS PACKAGE
    # ========================================================
    result = {
        "status": "success",
        "data": {
            "daily_forecast": dict(
                zip(
                    future_df["ds"].dt.strftime("%Y-%m-%d"),
                    future_df["Hybrid_future_corrected"].round(2),
                )
            ),
            "total_forecast": round(future_df["Hybrid_future_corrected"].sum(), 2),
            "mse": round(mse_corr, 2),
            "mae": round(mae_corr, 2),
            "mape": round(mape_corr, 2),
            "accuracy": round(100 - mape_corr, 2) if not np.isnan(mape_corr) else None,
        },
    }

    # Save history
    save_forecast_history(future_df)

    return result


# ============================================================
# 💾 HISTORY MANAGEMENT
# ============================================================
def save_forecast_history(future_df):
    """Append forecast results to CSV"""
    try:
        df_to_save = pd.DataFrame(
            {
                "Date": future_df["ds"].dt.strftime("%Y-%m-%d"),
                "Forecasted_Revenue": future_df["Hybrid_future_corrected"].round(2),
            }
        )
        if os.path.exists(HISTORY_FILE):
            df_to_save.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        else:
            df_to_save.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        print(f"⚠️ Could not save history: {e}")


# ============================================================
# 🌐 ROUTES
# ============================================================
@app.route("/api/revenue/forecast", methods=["POST"])
def api_forecast():
    try:
        result = generate_forecast()
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def api_download():
    """Download the most recent forecast"""
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({"status": "error", "message": "No forecast history yet"}), 404
        df = pd.read_csv(HISTORY_FILE)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="ForecastHistory")
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="Forecast_History.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/history", methods=["GET"])
def api_history():
    """Return forecast history"""
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify([])
        df = pd.read_csv(HISTORY_FILE)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# 🚀 MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
