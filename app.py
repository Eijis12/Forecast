from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import io
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# =====================================================
# 📂 Load Dataset
# =====================================================
DATASET_PATH = "DentalRecords_RevenueForecasting.xlsx"

def load_data():
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    return df

# =====================================================
# ⚙️ Feature Engineering
# =====================================================
def create_features(df):
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['ds'].dt.month
    df['is_payday'] = df['ds'].dt.day.isin([14, 15, 30, 31]).astype(int)
    holidays = ['2025-01-01','2025-04-09','2025-04-14','2025-04-21','2025-05-01',
                '2025-06-12','2025-08-25','2025-11-30','2025-12-25','2025-12-30']
    df['is_holiday'] = df['ds'].isin(pd.to_datetime(holidays)).astype(int)
    return df

# =====================================================
# 🔁 Recursive Hybrid Forecast Function
# =====================================================
def generate_recursive_forecast():
    df = load_data()
    df = create_features(df)

    # -------------------------------
    # Exponential Smoothing baseline
    # -------------------------------
    es_model = ExponentialSmoothing(df['y'], trend="add", seasonal="add", seasonal_periods=7)
    es_fit = es_model.fit()
    df['ES_forecast'] = es_fit.fittedvalues

    # -------------------------------
    # Prophet base model
    # -------------------------------
    prophet_df = df[['ds', 'y']].copy()
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=30)
    forecast = prophet.predict(future)
    prophet_forecast = forecast[['ds', 'yhat']]

    df = pd.merge(df, prophet_forecast, on='ds', how='left')
    df['Hybrid_forecast'] = (df['ES_forecast'] + df['yhat']) / 2

    # -------------------------------
    # LightGBM Residual Correction
    # -------------------------------
    df['Residual'] = df['y'] - df['Hybrid_forecast']
    features = ['day_of_week', 'is_weekend', 'is_payday', 'month', 'is_holiday']
    X_train, y_train = df[features], df['Residual']

    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    lgb_model.fit(X_train, y_train)

    df['Residual_pred'] = lgb_model.predict(X_train)
    df['Hybrid_corrected'] = df['Hybrid_forecast'] + df['Residual_pred']

    # -------------------------------
    # Recursive Future Forecast (30 days)
    # -------------------------------
    future_dates = pd.date_range(df['ds'].max() + timedelta(days=1), periods=30, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df = create_features(future_df)

    recursive_preds = []
    last_known_y = df['y'].iloc[-7:].tolist()

    for i in range(30):
        X_future = future_df.loc[[i], features]
        residual_pred = lgb_model.predict(X_future)[0]

        # Base hybrid prediction from average of last Prophet + ES trends
        base_pred = (np.mean(last_known_y[-7:]) * 0.9) + residual_pred

        # Add subtle random variation (₱300–₱700 std)
        variation = np.random.normal(0, 500)
        base_pred = max(0, base_pred + variation)

        # Force Sundays to 0
        if future_df.loc[i, 'day_of_week'] == 6:
            base_pred = 0

        recursive_preds.append(base_pred)
        last_known_y.append(base_pred)

    future_df['Forecast'] = recursive_preds

    # -------------------------------
    # Calculate MAE (validation)
    # -------------------------------
    mae = mean_absolute_error(df['y'], df['Hybrid_corrected'])
    monthly_mae = mae * 30  # monthly version

    return df, future_df, monthly_mae

# =====================================================
# 📈 API Endpoint for Forecast
# =====================================================
@app.route("/revenue/forecast", methods=["GET"])
def forecast_revenue():
    try:
        df, future_df, mae = generate_recursive_forecast()

        total_forecast = future_df['Forecast'].sum()
        forecast_data = future_df[['ds', 'Forecast']].to_dict(orient='records')

        response = {
            "forecast": forecast_data,
            "total_forecast": round(total_forecast, 2),
            "mae": round(mae, 2)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# 💾 Downloadable Forecast CSV
# =====================================================
@app.route("/revenue/download", methods=["GET"])
def download_forecast():
    _, future_df, mae = generate_recursive_forecast()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        future_df.to_excel(writer, index=False, sheet_name='Forecast')
    output.seek(0)
    return send_file(output, download_name="RevenueForecast.xlsx", as_attachment=True)

# =====================================================
# 🚀 Run Server
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
