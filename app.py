from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
import logging
import sys
import os
from datetime import datetime

# ===========================
# 🔧 Logging setup
# ===========================
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ===========================
# 🔧 Flask app
# ===========================
app = Flask(__name__)
CORS(app)

# ===========================
# 🔧 Load ML model (optional)
# ===========================
MODEL_PATH = "revenue_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Revenue model loaded successfully.")
    except Exception as e:
        logger.error(f"⚠️ Failed to load model: {e}")
else:
    logger.warning("⚠️ No model file found — using dummy forecast.")

# ===========================
# 🔹 Dummy forecast function
# ===========================
def generate_dummy_forecast():
    """Fallback if model not available."""
    today = datetime.today()
    days = pd.date_range(today, periods=7).strftime("%Y-%m-%d").tolist()
    revenue = np.random.randint(500, 5000, size=7).tolist()
    return list(zip(days, revenue))

# ===========================
# 🔹 Forecast API
# ===========================
@app.route("/api/revenue/forecast", methods=["GET"])
def forecast():
    try:
        logger.info("📈 Generating revenue forecast...")

        # If a model exists, attempt to generate prediction
        if model:
            # Dummy input — replace with your real features
            next_features = np.array([[1, 2, 3, 4]])
            predicted = model.predict(next_features)[0]
            response = {
                "status": "success",
                "forecast": [{"date": datetime.today().strftime("%Y-%m-%d"), "revenue": float(predicted)}]
            }
        else:
            # Use dummy data
            forecast_data = generate_dummy_forecast()
            response = {
                "status": "success",
                "forecast": [{"date": d, "revenue": r} for d, r in forecast_data]
            }

        logger.info("✅ Forecast generated successfully.")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ FORECAST ERROR: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================
# 🔹 Forecast history API (optional)
# ===========================
@app.route("/api/revenue/history", methods=["GET"])
def history():
    try:
        logger.info("📊 Fetching forecast history...")

        # Example dummy data
        history_data = [
            {"date": "2025-10-01", "revenue": 3200},
            {"date": "2025-10-02", "revenue": 4500},
            {"date": "2025-10-03", "revenue": 2800},
        ]

        return jsonify({"status": "success", "history": history_data}), 200

    except Exception as e:
        logger.error(f"❌ HISTORY ERROR: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================
# 🔹 Root route
# ===========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Forecast API is running."}), 200


# ===========================
# 🔧 Run app
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
