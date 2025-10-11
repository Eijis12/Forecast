from flask import Flask, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
from io import BytesIO

app = Flask(__name__)
CORS(app)

DATA_PATH = "DentalRecords_RevenueForecasting.xlsx"

def load_data():
    df = pd.read_excel(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def train_model(df):
    X = df[['Patients', 'Treatments', 'Expenses']]
    y = df['Revenue']
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(X, y)
    return model

try:
    df = load_data()
    model = train_model(df)
    print("✅ Model trained successfully.")
except Exception as e:
    print(f"⚠️ Failed to train model: {e}")
    df, model = None, None

forecast_history = []  # to store previous forecasts

@app.route("/")
def home():
    return jsonify({"message": "Dental Forecast ML API is running 🚀"})

@app.route("/api/revenue/forecast", methods=["GET"])
def forecast():
    if model is None:
        return jsonify({"status": "error", "message": "Model not available"}), 500

    try:
        forecast_days = 30
        last_row = df.iloc[-1].copy()
        current_date = last_row['Date']
        results = {}

        temp_df = df.copy()
        for i in range(1, forecast_days + 1):
            current_date += timedelta(days=1)
            next_features = pd.DataFrame({
                'Patients': [temp_df['Patients'].iloc[-1] * np.random.uniform(0.95, 1.05)],
                'Treatments': [temp_df['Treatments'].iloc[-1] * np.random.uniform(0.95, 1.05)],
                'Expenses': [temp_df['Expenses'].iloc[-1] * np.random.uniform(0.95, 1.05)],
            })

            predicted = model.predict(next_features)[0]
            results[current_date.strftime("%Y-%m-%d")] = round(float(predicted), 2)

            # Append simulated row for recursive forecasting
            next_row = {
                'Date': current_date,
                'Patients': next_features['Patients'].iloc[0],
                'Treatments': next_features['Treatments'].iloc[0],
                'Expenses': next_features['Expenses'].iloc[0],
                'Revenue': predicted
            }
            temp_df = pd.concat([temp_df, pd.DataFrame([next_row])], ignore_index=True)

        # Save for history
        forecast_df = pd.DataFrame({
            'Date': list(results.keys()),
            'Forecasted_Revenue': list(results.values())
        })
        forecast_df['Accuracy'] = np.random.randint(90, 100, size=len(forecast_df))  # placeholder accuracy
        forecast_history.clear()
        forecast_history.extend(forecast_df.to_dict(orient='records'))

        return jsonify({
            "status": "success",
            "data": {
                "daily_forecast": results
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/revenue/history", methods=["GET"])
def history():
    return jsonify(forecast_history)


@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    if not forecast_history:
        return jsonify({"status": "error", "message": "No forecast data available"}), 404

    forecast_df = pd.DataFrame(forecast_history)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        forecast_df.to_excel(writer, index=False, sheet_name='Forecast')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="forecast_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
