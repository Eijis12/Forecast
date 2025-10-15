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

# ===============================================================
# Helper: Calculate Metrics
# ===============================================================
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return mae, rmse, mape

# ===============================================================
# Route: Forecast Revenue
# ===============================================================
@app.route('/api/revenue/forecast', methods=['POST'])
def forecast_revenue():
    try:
        file_path = "DentalRecords_RevenueForecasting.xlsx"

        if not os.path.exists(file_path):
            return jsonify({"error": "Excel file not found."}), 400

        # Load data
        df = pd.read_excel(file_path)

        required_cols = {'YEAR', 'MONTH', 'DAY', 'REVENUE'}
        if not required_cols.issubset(df.columns):
            return jsonify({"error": "Excel must have columns: YEAR, MONTH, DAY, REVENUE"}), 400

        # Prepare time series data
        df['ds'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
        df = df.sort_values('ds')
        df['y'] = df['REVENUE']

        # Split train-test
        split_index = int(len(df) * 0.9)
        train, test = df.iloc[:split_index], df.iloc[split_index:]

        # ===============================================================
        # 1️⃣ Exponential Smoothing
        # ===============================================================
        es_model = ExponentialSmoothing(train['y'], trend='add', seasonal='add', seasonal_periods=12)
        es_fit = es_model.fit()
        es_forecast = es_fit.forecast(len(test))

        # ===============================================================
        # 2️⃣ Prophet Model
        # ===============================================================
        prophet_model = Prophet()
        prophet_model.fit(train[['ds', 'y']])
        future = prophet_model.make_future_dataframe(periods=len(test))
        prophet_forecast = prophet_model.predict(future)
        prophet_forecast = prophet_forecast.tail(len(test))['yhat'].values

        # ===============================================================
        # 3️⃣ LightGBM Model
        # ===============================================================
        train['dayofweek'] = train['ds'].dt.dayofweek
        train['month'] = train['ds'].dt.month
        test['dayofweek'] = test['ds'].dt.dayofweek
        test['month'] = test['ds'].dt.month

        X_train = train[['dayofweek', 'month']]
        y_train = train['y']
        X_test = test[['dayofweek', 'month']]
        y_test = test['y']

        lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=200)
        lgb_model.fit(X_train, y_train)
        lgb_forecast = lgb_model.predict(X_test)

        # ===============================================================
        # 🔹 Hybrid Model (Average of 3 Models)
        # ===============================================================
        hybrid_forecast = (es_forecast + prophet_forecast + lgb_forecast) / 3

        # ===============================================================
        # Calculate Metrics
        # ===============================================================
        mae, rmse, mape = calculate_metrics(y_test, hybrid_forecast)

        # ===============================================================
        # Forecast Next 30 Days
        # ===============================================================
        last_date = df['ds'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)

        # Prophet for future baseline
        future_prophet = prophet_model.make_future_dataframe(periods=30)
        prophet_future_pred = prophet_model.predict(future_prophet).tail(30)['yhat'].values

        # LightGBM for future
        future_df = pd.DataFrame({
            'ds': future_dates,
            'dayofweek': future_dates.dayofweek,
            'month': future_dates.month
        })
        lgb_future_pred = lgb_model.predict(future_df[['dayofweek', 'month']])

        # Exponential Smoothing for future
        es_future_pred = es_fit.forecast(30)

        # Hybrid future forecast
        hybrid_future_forecast = (prophet_future_pred + lgb_future_pred + es_future_pred) / 3

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Revenue': hybrid_future_forecast
        })

        # ===============================================================
        # Save to Excel
        # ===============================================================
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            forecast_df.to_excel(writer, index=False, sheet_name='Forecast')
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE'],
                'Value': [mae, rmse, mape]
            })
            metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
        output_excel.seek(0)

        return send_file(
            output_excel,
            as_attachment=True,
            download_name='Revenue_Forecast.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===============================================================
# Route: Forecast History
# ===============================================================
@app.route('/api/revenue/history', methods=['GET'])
def forecast_history():
    try:
        if not os.path.exists("Revenue_Forecast_History.xlsx"):
            return jsonify([])

        df = pd.read_excel("Revenue_Forecast_History.xlsx")
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================================================
# Route: Download Forecast
# ===============================================================
@app.route('/api/revenue/download', methods=['GET'])
def download_forecast():
    try:
        if not os.path.exists("Revenue_Forecast.xlsx"):
            return jsonify({"error": "No forecast file found"}), 404

        return send_file(
            "Revenue_Forecast.xlsx",
            as_attachment=True,
            download_name='Revenue_Forecast.xlsx'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================================================
# Root Route
# ===============================================================
@app.route('/')
def home():
    return "Revenue Forecast API is running with Hybrid Model (ES + Prophet + LightGBM)!"


# ===============================================================
# Run Flask App
# ===============================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
