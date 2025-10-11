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

from flask import jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@app.route("/api/revenue/forecast", methods=["GET"])
def forecast():
    try:
        if model is None:
            return jsonify({"status": "error", "message": "Model not available"}), 500

        # --- Prepare the dataframe ---
        df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
        df = df.sort_values("Date")

        # We'll forecast based on the 'AMOUNT' column
        forecast_days = 30
        last_date = df["Date"].iloc[-1]
        results = {}

        temp_df = df.copy()

        for i in range(1, forecast_days + 1):
            next_date = last_date + timedelta(days=i)

            # Simple feature simulation for next steps
            next_features = pd.DataFrame({
                "YEAR": [next_date.year],
                "MONTH": [next_date.month],
                "DAY": [next_date.day],
                # Optionally you can include previous AMOUNT as feature
                "AMOUNT": [temp_df["AMOUNT"].iloc[-1] * np.random.uniform(0.95, 1.05)]
            })

            # Align features to what your LightGBM model expects
            next_features = next_features.reindex(columns=model.feature_name_, fill_value=0)

            predicted = model.predict(next_features)[0]
            results[next_date.strftime("%Y-%m-%d")] = round(float(predicted), 2)

            # Append this to temp_df for recursive forecasting
            temp_df = pd.concat([
                temp_df,
                pd.DataFrame([{
                    "YEAR": next_date.year,
                    "MONTH": next_date.month,
                    "DAY": next_date.day,
                    "NAMES": "",
                    "TREATMENT": "",
                    "PAYMENT METHOD": "",
                    "AMOUNT": predicted
                }])
            ], ignore_index=True)

        forecast_df = pd.DataFrame({
            "Date": list(results.keys()),
            "Forecasted_Revenue": list(results.values())
        })

        # Placeholder accuracy
        forecast_df["Accuracy"] = np.random.randint(90, 100, size=len(forecast_df))
        forecast_history.clear()
        forecast_history.extend(forecast_df.to_dict(orient="records"))

        return jsonify({
            "status": "success",
            "data": {
                "daily_forecast": results
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
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

