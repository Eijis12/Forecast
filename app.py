from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATA_FILE = "DentalRecords_RevenueForecasting.xlsx"
MODEL_FILE = "trained_model.pkl"
HISTORY_FILE = "forecast_history.csv"


# ---------- Helper: Load or train model ----------
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"{DATA_FILE} not found.")

        df = pd.read_excel(DATA_FILE)

        # Auto-fix: Convert months if text-based
        if 'Month' in df.columns:
            try:
                df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            except:
                df['Month'] = pd.to_datetime(df['Month'], format='%B', errors='coerce')
        else:
            raise ValueError("Missing 'Month' column in dataset.")

        df = df.dropna(subset=['Month'])
        df['Month_Num'] = df['Month'].dt.month

        if 'Revenue' not in df.columns:
            raise ValueError("Missing 'Revenue' column in dataset.")

        X = df[['Month_Num']]
        y = df['Revenue']

        model = lgb.LGBMRegressor()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model


model = load_or_train_model()


# ---------- Route: Generate Forecast ----------
@app.route('/api/revenue/forecast', methods=['POST'])
def generate_forecast():
    try:
        future_months = np.arange(1, 13).reshape(-1, 1)
        predictions = model.predict(future_months)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "month": datetime(2025, i + 1, 1).strftime("%B"),
                "revenue": float(pred)
            })

        # Save history
        df_hist = pd.DataFrame(results)
        df_hist['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(HISTORY_FILE):
            df_hist.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        else:
            df_hist.to_csv(HISTORY_FILE, index=False)

        return jsonify({"status": "success", "forecast": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- Route: Forecast History ----------
@app.route('/api/revenue/history', methods=['GET'])
def get_forecast_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])
    df = pd.read_csv(HISTORY_FILE)
    records = df.to_dict(orient='records')
    return jsonify(records)


# ---------- Render-compatible startup ----------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
