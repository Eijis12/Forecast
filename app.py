from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import lightgbm as lgb
from datetime import timedelta

# ----------------------------------
# Flask setup
# ----------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------
# Load data and train model
# ----------------------------------
DATA_PATH = "DentalRecords_RevenueForecasting.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Training data
    X = df[['Patients', 'Treatments', 'Expenses']]
    y = df['Revenue']

    model = lgb.LGBMRegressor()
    model.fit(X, y)
    print("✅ Model trained successfully on DentalRecords_RevenueForecasting.xlsx")

except Exception as e:
    print(f"⚠️ Error loading dataset or training model: {e}")
    model = None


# ----------------------------------
# Routes
# ----------------------------------

@app.route("/")
def home():
    return jsonify({"message": "Dental Forecast ML API is running 🚀"})


@app.route("/api/revenue/forecast", methods=["GET"])
def forecast():
    """Generate 30-day revenue forecast."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not trained properly"}), 500

    try:
        last_row = df.iloc[-1]
        forecasts = []
        current_date = last_row['Date']

        for i in range(30):
            current_date += timedelta(days=1)

            # Simulated variations
            patients = max(0, last_row['Patients'] * (1 + 0.01 * (i % 5 - 2)))
            treatments = max(0, last_row['Treatments'] * (1 + 0.02 * (i % 3 - 1)))
            expenses = max(0, last_row['Expenses'] * (1 + 0.015 * (i % 4 - 2)))

            prediction = model.predict([[patients, treatments, expenses]])[0]

            forecasts.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_revenue": round(float(prediction), 2)
            })

        return jsonify({
            "status": "success",
            "data": forecasts
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------
# Render entry point
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
