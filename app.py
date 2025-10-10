from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from datetime import timedelta, datetime
import traceback
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Paths
BASE_DIR = os.path.dirname(__file__)
REVENUE_FILE = os.path.join(BASE_DIR, "DentalRecords_RevenueForecasting.xlsx")
MODEL_FILE = os.path.join(BASE_DIR, "lgb_revenue.pkl")

# Minimal training settings
LAGS = [1,2,3,7,14,28]
ROLL_WINDOWS = [7,14,30]
FORECAST_STEPS_DEFAULT = 30

# Utility: prepare daily series from file
def load_daily_series(file_path=REVENUE_FILE):
    df = pd.read_excel(file_path)
    # find amount column (case-insensitive)
    amount_cols = [c for c in df.columns if 'amount' in c.lower()]
    if not amount_cols:
        raise ValueError("No column with 'amount' found in dataset.")
    amount_col = amount_cols[0]

    # Build DATE if needed
    if 'DATE' not in df.columns:
        if all(col in df.columns for col in ['YEAR','MONTH','DAY']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-' + df['DAY'].astype(str))
        elif all(col in df.columns for col in ['YEAR','MONTH']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        else:
            raise ValueError("No DATE or YEAR/MONTH(DAY) columns found in dataset.")

    df = df.dropna(subset=['DATE'])
    df['DATE'] = pd.to_datetime(df['DATE'])
    daily = df.groupby('DATE')[amount_col].sum().sort_index()
    # make continuous daily index
    daily = daily.asfreq('D').fillna(0)
    daily.name = 'y'
    return daily

# Feature builder for training set (returns DataFrame with features and target 'y')
def build_feature_table(series: pd.Series):
    df = pd.DataFrame(series).rename(columns={series.name:'y'})
    df = df.sort_index()

    # Date parts
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['is_sunday'] = (df.index.dayofweek == 6).astype(int)

    # Lags
    for lag in LAGS:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Rolling stats
    for w in ROLL_WINDOWS:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].shift(1).rolling(w).std().fillna(0)

    # drop rows with NaNs caused by shifts
    df = df.dropna()

    # features list (exclude y)
    features = [c for c in df.columns if c != 'y']
    return df, features

# Train LightGBM model (returns trained booster and features list)
def train_model(series: pd.Series):
    if len(series) < 14:
        # Not enough data — do not train
        return None, None

    df_feat, features = build_feature_table(series)
    X = df_feat[features]
    y = df_feat['y']

    # time-based train/val split (last 30 days val if available)
    val_days = min(30, int(len(X)*0.2))
    if val_days < 7: val_days = int(len(X)*0.2)
    split_idx = len(X) - val_days if val_days > 0 else len(X)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'feature_pre_filter': False
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Save model
    try:
        joblib.dump({'model': booster, 'features': features}, MODEL_FILE)
    except Exception:
        pass

    return booster, features

# Helper: build features for a single next date, using current series
def build_features_for_next_date(cur_series: pd.Series, next_date: pd.Timestamp, features_list):
    feat = {}
    feat['dow'] = next_date.dayofweek
    feat['month'] = next_date.month
    feat['day'] = next_date.day
    feat['is_sunday'] = 1 if next_date.weekday() == 6 else 0

    # lags
    for lag in LAGS:
        idx = -lag
        if len(cur_series) >= lag:
            feat[f'lag_{lag}'] = float(cur_series.iloc[idx])
        else:
            feat[f'lag_{lag}'] = float(cur_series.mean())

    # rolling
    for w in ROLL_WINDOWS:
        if len(cur_series) >= 1:
            feat[f'roll_mean_{w}'] = float(cur_series.iloc[-w:].mean()) if len(cur_series) >= 1 else float(cur_series.mean())
            feat[f'roll_std_{w}'] = float(cur_series.iloc[-w:].std()) if len(cur_series) >= 2 else 0.0
        else:
            feat[f'roll_mean_{w}'] = 0.0
            feat[f'roll_std_{w}'] = 0.0

    # Ensure ordering of keys as features_list
    return [feat.get(f, 0.0) for f in features_list]

# Recursive forecast using the trained LightGBM model
def recursive_forecast(model, features_list, historical_series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    cur = historical_series.copy()
    out = []

    for i in range(steps):
        next_date = cur.index[-1] + timedelta(days=1)

        # Sundays closed: revenue = 0
        if next_date.weekday() == 6:
            yhat = 0.0
        else:
            X_row = build_features_for_next_date(cur, next_date, features_list)
            try:
                yhat = float(model.predict([X_row])[0])
            except Exception:
                # prediction issue -> fallback to last value
                yhat = float(cur.iloc[-1])

        # prevent negative
        yhat = max(0.0, yhat)
        cur.loc[next_date] = yhat
        out.append((next_date, yhat))

    # Convert to dict with string dates
    forecast_dict = {d.strftime("%Y-%m-%d"): round(float(v),2) for d,v in out}
    total = sum(v for _,v in out)
    # quick approximate CI: +/- 12%
    conf_lower = {k: round(v*0.88,2) for k,v in forecast_dict.items()}
    conf_upper = {k: round(v*1.12,2) for k,v in forecast_dict.items()}

    return total, forecast_dict, conf_lower, conf_upper

# Fallback fast average method (if not enough data or model missing)
def fast_fallback_forecast(series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    last = float(series.dropna().iloc[-1]) if len(series.dropna())>0 else 0.0
    # use 7-day mean if possible
    if len(series) >= 7:
        base = float(series.iloc[-7:].mean())
    else:
        base = last if last>0 else float(series.mean() if len(series)>0 else 0.0)
    # create small upward trend
    forecast_vals = np.linspace(base, base*1.03, steps)
    dates = [ (series.index[-1] + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(steps)]
    forecast_dict = {dates[i]: round(float(forecast_vals[i]),2) for i in range(steps)}
    total = float(sum(forecast_vals))
    conf_lower = {k: round(v*0.9,2) for k,v in forecast_dict.items()}
    conf_upper = {k: round(v*1.1,2) for k,v in forecast_dict.items()}
    # zero out Sundays
    for i, d in enumerate(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        if dt.weekday() == 6:
            forecast_dict[d]=0.0
            conf_lower[d]=0.0
            conf_upper[d]=0.0
    return total, forecast_dict, conf_lower, conf_upper

# Load data and train on startup (attempt to load model file if exists)
GLOBAL_MODEL = None
GLOBAL_FEATURES = None
GLOBAL_DAILY = None

def startup_load_and_train():
    global GLOBAL_MODEL, GLOBAL_FEATURES, GLOBAL_DAILY
    print("🔁 Loading data and initializing model...")

    # load series
    try:
        daily = load_daily_series(REVENUE_FILE)
    except Exception as e:
        print("❌ Failed to load revenue file:", e)
        GLOBAL_DAILY = None
        return

    GLOBAL_DAILY = daily

    # try to load saved model
    if os.path.exists(MODEL_FILE):
        try:
            saved = joblib.load(MODEL_FILE)
            GLOBAL_MODEL = saved.get('model')
            GLOBAL_FEATURES = saved.get('features')
            print("✅ Loaded saved model from disk.")
            return
        except Exception:
            print("⚠️ Failed to load saved model, will retrain.")

    # train new one
    try:
        start = time.time()
        model, features = train_model(daily)
        elapsed = time.time() - start
        if model is not None:
            GLOBAL_MODEL = model
            GLOBAL_FEATURES = features
            print(f"✅ Trained LightGBM model in {elapsed:.2f}s. Features:", len(features))
        else:
            print("⚠️ Not enough data to train LightGBM, will use fallback.")
    except Exception as e:
        print("❌ Training failed:", e)
        traceback.print_exc()
        GLOBAL_MODEL = None
        GLOBAL_FEATURES = None

# Call once at startup
startup_load_and_train()


@app.route("/")
def home():
    return jsonify({"message": "Revenue forecasting service is UP"})

@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    """
    Returns:
    {
      status: "success",
      data: {
        next_month_total: float,
        daily_forecast: { "YYYY-MM-DD": value, ... },
        confidence_intervals: { "lower": {...}, "upper": {...} }
      }
    }
    """
    start_t = time.time()
    try:
        # reload data up to today (so if file updated, we use latest)
        daily = load_daily_series(REVENUE_FILE)
        # ensure daily available
        if daily is None or len(daily)==0:
            raise ValueError("No daily revenue data found.")

        # Prefer trained model; if not available, fallback
        if GLOBAL_MODEL is not None and GLOBAL_FEATURES is not None:
            total, forecast_dict, conf_lower, conf_upper = recursive_forecast(
                GLOBAL_MODEL, GLOBAL_FEATURES, daily, steps=FORECAST_STEPS_DEFAULT
            )
        else:
            total, forecast_dict, conf_lower, conf_upper = fast_fallback_forecast(daily, steps=FORECAST_STEPS_DEFAULT)

        elapsed = time.time() - start_t
        print(f"✅ Forecast computed in {elapsed:.2f}s")
        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": {
                "next_month_total": round(total,2),
                "daily_forecast": forecast_dict,
                "confidence_intervals": {"lower": conf_lower, "upper": conf_upper},
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        })
    except Exception as e:
        print("❌ Forecast failed:")
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
