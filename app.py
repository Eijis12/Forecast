import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from lightgbm import LGBMRegressor
import joblib

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

MODEL_PATH = "revenue_model.pkl"
FORECAST_HISTORY_PATH = "forecast_history.csv"
DATA_PATH = "DentalRecords_RevenueForecasting.xlsx"

# ---------- Utility Functions ----------

def load_data():
    """Load and clean dataset, handling both month names and dates."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found in directory.")

    df = pd.read_excel(DATA_PATH)

    # Try to detect a usable date column
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            # fallback if contains text months
            df['Date'] = pd.to_datetime(df['Date'] + ' 2025', format='%B %Y')
    elif 'Month' in df.columns:
        # Example: "January", "February"
        df['Date'] = pd.to_datetime(df['Month'] + ' 2025', format='%B %Y')
    else:
        raise ValueError("No 'Date' or 'Month' column found in dataset.")

    # Try to find the revenue column
    revenue_col = None
    for col in df.columns:
        if 'revenue' in col.lower() or 'income' in col.lower() or 'sales' in col.lower():
            revenue_col = col
            break
    if not revenue_col:
        raise ValueError("No column found for revenue/income/sales.")

    df = df[['Date', revenue_col]].rename(columns={revenue_col: 'Revenue'})
    df = df.sort_values('Date')
    df = df.set_index('Date')
    return df


def train_model():
    """Train and save the LightGBM model."""
    df = load_data()

    # Create features
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['prev_revenue'] = df['Revenue'].shift(1)
    df = df.dropna()

    X = df[['month', 'year', 'prev_revenue']]
    y = df['Revenue']

    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    """Load existing model or train a new one if missing."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model()


def save_forecast_to_history(forecast_value):
    """Save forecast result with timestamp to history file."""
    history_entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Forecasted Revenue': forecast_value
    }])

    if os.path.exists(FORECAST_HISTORY_PATH):
        history = pd.read_csv(FORECAST_HISTORY_PATH)
        history = pd.concat([history, history_entry], ignore_index=True)
    else:
        history = history_entry

    history.to_csv(FORECAST_HISTORY_PATH, index=False)


def get_forecast_history():
    """Read forecast history CSV if it exists."""
    if os.path.exists(FORECAST_HISTORY_PATH):
        return pd.read_csv(FORECAST_HISTORY_PATH).to_dict(orient="records")
    return []

# ---------- Routes ----------

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/revenue/forecast', methods=['POST'])
def generate_forecast():
    try:
        model = load_model()
        df = load_data()

        last_row = df.iloc[-1]
        next_month = last_row.name.month + 1
        next_year = last_row.name.year
        if next_month > 12:
            next_month = 1
            next_year += 1

        X_new = pd.DataFrame([{
            'month': next_month,
            'year': next_year,
            'prev_revenue': last_row['Revenue']
        }])

        forecast = model.predict(X_new)[0]
        save_forecast_to_history(float(forecast))

        return jsonify({
            "message": "Forecast generated successfully!",
            "forecast": float(forecast)
        })

    except Exception as e:
        print("Error generating forecast:", e)
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route('/api/revenue/history', methods=['GET'])
def get_history():
    try:
        data = get_forecast_history()
        return jsonify(data)
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"}), 500


# ---------- Appointments & Users (Dummy Endpoints for UI) ----------

@app.route('/api/users', methods=['POST'])
def create_user():
    return jsonify({"message": "User created successfully!"})


@app.route('/api/appointments', methods=['GET', 'POST'])
def appointments():
    return jsonify({"message": "Appointments endpoint placeholder."})


# ---------- Run App ----------

if __name__ == '__main__':
    app.run(debug=True)
