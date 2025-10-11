from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import io
import datetime
import traceback
import joblib
from lightgbm import LGBMRegressor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------- File paths ----------
DATA_FILE = "DentalRecords_RevenueForecasting.xlsx"
MODEL_FILE = "trained_model.pkl"
HISTORY_FILE = "forecast_history.csv"


# ---------- Helper: Load or train model ----------
def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILE):
        print("✅ Loaded existing model from disk.")
        return joblib.load(MODEL_FILE)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    df = pd.read_excel(DATA_FILE)
    df.columns = [col.strip().lower() for col in df.columns]

    if "month" not in df.columns or "amount" not in df.columns:
        raise ValueError(f"Missing required columns. Found: {', '.join(df.columns)}")

    # Convert month text to numbers
    df["month_num"] = pd.to_datetime(df["month"], errors="coerce", format="%B").dt.month
    if df["month_num"].isna().any():
        df["month_num"] = pd.to_datetime(df["month"], errors="coerce", format="%b").dt.month

    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)

    X = df[["month_num"]]
    y = df["amount"]

    model = LGBMRegressor()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("✅ Trained and saved new model.")
    return model


# ---------- Initialize model ----------
model = load_or_train_model()


# ---------- Route: Generate Forecast ----------
@app.route("/api/revenue/forecast", methods=["POST"])
def forecast_revenue():
    try:
        excel_path = "DentalRecords_RevenueForecasting.xlsx"

        if not os.path.exists(excel_path):
            return jsonify({"status": "error", "message": "Revenue dataset not found."}), 404

        df = pd.read_excel(excel_path)

        # ✅ Ensure column consistency
        if "MONTH" in df.columns:
            df.rename(columns={"MONTH": "Date"}, inplace=True)
        if "REVENUE" in df.columns:
            df.rename(columns={"REVENUE": "Revenue"}, inplace=True)

        # ✅ Convert to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Revenue"]).sort_values("Date")

        # ✅ Prepare the training data
        df["Day"] = (df["Date"] - df["Date"].min()).dt.days
        X = df[["Day"]]
        y = df["Revenue"]

        model = lgb.LGBMRegressor()
        model.fit(X, y)

        # ✅ Forecast for the next 30 days from TODAY
        today = datetime.datetime.now()
        future_dates = pd.date_range(today, today + datetime.timedelta(days=30), freq="D")

        # Map to numeric "Day" index continuing from last
        last_day = df["Day"].max()
        future_days = np.arange(last_day + 1, last_day + len(future_dates) + 1).reshape(-1, 1)

        # ✅ Predict
        future_forecast = model.predict(future_days)

        forecast_df = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Forecasted_Revenue": future_forecast.round(2)
        })

        # ✅ Save to Excel for download
        output_path = "forecast_results.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            forecast_df.to_excel(writer, index=False, sheet_name="Forecast")

        # ✅ Save history to memory (optional for your /api/revenue/history route)
        forecast_df.to_csv("forecast_history.csv", index=False)

        return jsonify({
            "status": "success",
            "message": "30-day real-time forecast generated.",
            "data": forecast_df.to_dict(orient="records")
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error generating forecast: {str(e)}"
        }), 500



# ---------- Route: Forecast History ----------
@app.route("/api/revenue/history", methods=["GET"])
def get_forecast_history():
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify([])
        df = pd.read_csv(HISTORY_FILE)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        print("❌ History load error:", e)
        return jsonify([])


# ---------- Route: Download Forecast ----------
@app.route("/api/revenue/download", methods=["GET"])
def download_forecast():
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({"status": "error", "message": "No forecast data available"}), 404

        df = pd.read_csv(HISTORY_FILE)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast")
        output.seek(0)
        return send_file(output, download_name="forecast_results.xlsx", as_attachment=True)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- Root ----------
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Revenue Forecast API is running."})


# ---------- Start ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

