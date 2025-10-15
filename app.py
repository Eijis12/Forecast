from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import datetime
from prophet import Prophet
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================================
# 📘 Utility Functions
# ==========================================================
def prepare_dataframe(file_path="DentalRecords_RevenueForecasting.xlsx"):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    if not {'year', 'month', 'day', 'revenue'}.issubset(df.columns):
        raise ValueError("Excel must have columns: YEAR, MONTH, DAY, REVENUE")

    df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df[['ds', 'revenue']].rename(columns={'revenue': 'y'})
    df = df.sort_values('ds')
    return df


def calculate_metrics(actual, predicted):
    # Align lengths
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
    return mae, rmse, mape


# ==========================================================
# 🧠 Hybrid Model (Exponential Smoothing + Prophet + LightGBM)
# ==========================================================
def hybrid_forecast(df, forecast_days=30):
    df = df.copy()
    train = df.copy()

    # --- Exponential Smoothing ---
    es_model = ExponentialSmoothing(
        train['y'], trend='add', seasonal='add', seasonal_periods=7
    ).fit()
    es_forecast = es_model.forecast(forecast_days)

    # --- Prophet ---
    prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(train)
    future = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future)
    prophet_values = prophet_forecast.tail(forecast_days)['yhat'].values

    # --- LightGBM ---
    train['dayofweek'] = train['ds'].dt.dayofweek
    train['month'] = train['ds'].dt.month
    train['year'] = train['ds'].dt.year
    features = ['dayofweek', 'month', 'year']

    X = train[features]
    y = train['y']
    lgb_train = lgb.Dataset(X, label=y)
    params = {'objective': 'regression', 'verbosity': -1, 'boosting_type': 'gbdt'}
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

    # Create next 30 days
    last_date = df['ds'].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_features = pd.DataFrame({
        'dayofweek': [d.weekday() for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates]
    })

    lgb_forecast = lgb_model.predict(future_features)

    # --- Combine results ---
    combined_forecast = (es_forecast.values + prophet_values + lgb_forecast) / 3

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Revenue': combined_forecast
    })

    # --- Make Sundays (dayofweek=6 in Python, but 0 in JS) zero ---
    forecast_df.loc[[d.weekday() == 6 for d in forecast_df['Date']], 'Forecasted_Revenue'] = 0

    total_forecast = forecast_df['Forecasted_Revenue'].sum()

    # --- Backtest metrics (on last N days of actuals) ---
    backtest_size = min(30, len(df))
    actual_recent = df['y'].tail(backtest_size).values
    predicted_recent = es_model.fittedvalues.tail(backtest_size).values
    mae, rmse, mape = calculate_metrics(actual_recent, predicted_recent)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    return forecast_df, total_forecast, metrics


# ==========================================================
# 📈 API Routes
# ==========================================================
@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_revenue():
    try:
        df = prepare_dataframe()
        forecast_df, total_forecast, metrics = hybrid_forecast(df)

        result = {
            "status": "success",
            "data": {
                "daily_forecast": dict(zip(forecast_df['Date'].dt.strftime('%Y-%m-%d'), forecast_df['Forecasted_Revenue'])),
                "total_forecast": float(total_forecast),
                "metrics": metrics
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        df = prepare_dataframe()
        forecast_df, total_forecast, metrics = hybrid_forecast(df)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            forecast_df.to_excel(writer, index=False, sheet_name='Forecast')
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="Forecast_Output.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "Revenue Forecast API with Hybrid Model + Metrics"})


# ==========================================================
# 🏁 Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
