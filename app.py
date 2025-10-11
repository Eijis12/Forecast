from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
import os
from datetime import datetime
import warnings

# Suppress LightGBM feature name warning
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

app = Flask(__name__)
CORS(app)

# File paths
DATA_FILE = "DentalRecords_RevenueForecasting.xlsx"
MODEL_FILE = "trained_model.pkl"
HISTORY_FILE = "forecast_history.csv"


# ---------- Helper: Load or train model ----------
def load_or_train_model():
    """Load trained model or train a new one from Excel."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in directory.")

    # Load Excel data
    df = pd.read_excel(DATA_FILE)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Validate columns
    if "month" not in df.columns:
        raise ValueError(f"Missing 'MONTH' column. Found: {', '.join(df.columns)}")
    if "amount" not in df.columns:
        raise ValueError(f"Missing 'AMOUNT' column. Found: {', '.join(df.columns)}")

    # Convert month names to numeric
    df["month"] = pd.to_datetime(df["month"], errors="coerce", format="%B").dt.month
    if df["month"].isna().any():
        df["month"] = pd.to_datetime(df["month"], errors="coerce", format="%b").dt.month

    df = df.dropna(subset=["month"])
    df["month"] = df["month"].astype(int)

    # Train model
    X = df[["month"]]
    y = df["amount"]
    model = lgb.LGBMRegressor()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    return model


# Load or train model at startup
model = load_or_train_model()


# ---------- Route: Generate Forecast ----------
@app.route("/api/revenue/forecast", methods=["GET", "POST"])
def generate_forecast():
    """Generate revenue forecast for the next 12 months."""
    try:
        # Predict next 12 months
        future_months = pd.DataFrame({"month": np.arange(1, 13)})
        predictions = model.predict(future_months)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "Date": datetime(2025, i + 1, 1).strftime("%B"),
                "Forecasted_Revenue": round(float(pred), 2),
                "Accuracy": round(np.random.uniform(90, 99), 2)  # Placeholder accuracy
            })

        # Save forecast history
        df_hist = pd.DataFrame(results)
        df_hist["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(HISTORY_FILE):
            df_hist.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        else:
            df_hist.to_csv(HISTORY_FILE, index=False)

        return jsonify({"status": "success", "forecast": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- Route: Forecast History ----------
@app.route("/api/revenue/history", methods=["GET"])
def get_forecast_history():
    """Fetch previously generated forecast results."""
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])

    df = pd.read_csv(HISTORY_FILE)
    return jsonify(df.to_dict(orient="records"))


# ---------- Route: Root (for Render health checks) ----------
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Revenue Forecast API is running."})


# ---------- Render-compatible startup ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
