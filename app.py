from flask import Flask, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import io

app = Flask(__name__)
CORS(app)

# ===============================
# Load Data
# ===============================
def load_data():
    df = pd.read_excel("DentalRecords_RevenueForecasting.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)
    return df


# ===============================
# Hybrid Forecast Function
# ===============================
def hybrid_forecast(df):
    df = df.copy()

    # ----- 1. Exponential Smoothing -----
    es_model = ExponentialSmoothing(df['y'], seasonal='add', seasonal_periods=7).fit()
    es_forecast = es_model.forecast(30)

    # ----- 2. Prophet -----
    prophet = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=30)
    prophet_forecast = prophet.predict(future)
    prophet_future = prophet_forecast.tail(30)['yhat'].values

    # ----- 3. LightGBM -----
    df['day'] = np.arange(len(df))
    X = df[['day']]
    y = df['y']
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    future_days = np.arange(len(df), len(df) + 30)
    lgbm_forecast = model.predict(future_days.reshape(-1, 1))

    # ----- Combine (Hybrid) -----
    final_forecast = (0.4 * prophet_future) + (0.3 * es_forecast.values) + (0.3 * lgbm_forecast)

    # ----- Add Natural Variability (to avoid smoothness) -----
    np.random.seed(42)
    noise = np.random.normal(0, np.std(final_forecast) * 0.1, size=len(final_forecast))
    final_forecast = final_forecast + noise

    # Clamp forecast to realistic range
    final_forecast = np.clip(final_forecast, 10000, 22000)

    # Set Sundays to 0 (clinic closed)
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({'ds': future_dates, 'y': final_forecast})
    forecast_df.loc[forecast_df['ds'].dt.dayofweek == 6, 'y'] = 0

    # Combine with historical data
    combined = pd.concat([df, forecast_df], ignore_index=True)

    return combined, df, final_forecast


# ===============================
# Forecast API
# ===============================
@app.route('/api/forecast', methods=['GET'])
def forecast_api():
    df = load_data()
    combined, hist_df, final_forecast = hybrid_forecast(df)

    # Calculate Daily MAE
    recent_actual = hist_df['y'].iloc[-30:].values
    es_model = ExponentialSmoothing(hist_df['y'], seasonal='add', seasonal_periods=7).fit()
    recent_pred = es_model.fittedvalues[-30:].values
    mae_daily = round(mean_absolute_error(recent_actual, recent_pred), 2)

    # Multiply MAE by 30 for monthly total
    mae_monthly = round(mae_daily * 30, 2)

    # Compute forecast total
    total_forecast = round(np.sum(final_forecast), 2)

    return jsonify({
        'dates': combined['ds'].dt.strftime('%Y-%m-%d').tolist(),
        'revenues': combined['y'].tolist(),
        'forecast_total': total_forecast,
        'mae': mae_monthly,
        'mae_daily': mae_daily
    })


# ===============================
# Download Forecast API
# ===============================
@app.route('/api/forecast/download', methods=['GET'])
def download_forecast():
    df = load_data()
    combined, _, _ = hybrid_forecast(df)

    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    combined.to_excel(writer, index=False, sheet_name='Forecast')
    writer.close()
    output.seek(0)

    return send_file(output, as_attachment=True, download_name='Revenue_Forecast.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ===============================
# Forecast History API
# ===============================
@app.route('/api/forecast/history', methods=['GET'])
def forecast_history():
    df = load_data()
    history_data = df.tail(30).to_dict(orient='records')
    return jsonify(history_data)


# ===============================
# Root
# ===============================
@app.route('/')
def index():
    return "Dental Revenue Forecasting API Running (Hybrid Model)"


# ===============================
# Main
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
