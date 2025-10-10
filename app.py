from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from datetime import timedelta, datetime, date
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
TRAIN_WINDOW_DAYS = 365   # use last 12 months for training
OUTLIER_QUANTILE = 0.95  # cap outliers at 95th percentile
SUNDAY_WEEKDAY = 6       # Monday=0 .. Sunday=6

# ------------------------------
# Utility: load daily series
# ------------------------------
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

    # continuous daily index (fill missing days with 0)
    daily = daily.asfreq('D').fillna(0)
    daily.name = 'y'
    return daily

# ------------------------------
# Preprocess / cap / window
# ------------------------------
def prepare_series_for_training(raw_series: pd.Series, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS):
    s = raw_series.copy().sort_index()
    # cap outliers
    cap = float(s.quantile(cap_quantile))
    s = s.clip(upper=cap)

    # set Sundays to 0 (clinic closed)
    s[s.index.weekday == SUNDAY_WEEKDAY] = 0.0

    # restrict to last N days
    if len(s) > 0:
        cutoff = s.index.max() - pd.Timedelta(days=days_window)
        s = s[s.index >= cutoff]

    s.name = 'y'
    return s

# ------------------------------
# Feature building
# ------------------------------
def build_feature_table(series: pd.Series):
    df = pd.DataFrame(series).rename(columns={series.name:'y'})
    df = df.sort_index()

    # Date parts
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['is_sunday'] = (df.index.dayofweek == SUNDAY_WEEKDAY).astype(int)

    # Lags
    for lag in LAGS:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Rolling stats
    for w in ROLL_WINDOWS:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].shift(1).rolling(w).std().fillna(0)

    df = df.dropna()
    features = [c for c in df.columns if c != 'y']
    return df, features

# ------------------------------
# Train LightGBM
# ------------------------------
def train_model(series: pd.Series):
    series = series.copy()
    series.name = 'y'
    if len(series) < 14:
        return None, None

    df_feat, features = build_feature_table(series)
    X = df_feat[features]
    y = df_feat['y']

    # time-based split
    val_days = min(30, int(len(X)*0.2))
    if val_days < 7:
        val_days = int(len(X)*0.2)
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

    # Save model and feature list
    try:
        joblib.dump({'model': booster, 'features': features}, MODEL_FILE)
    except Exception:
        pass

    return booster, features

# ------------------------------
# Build features for a next date
# ------------------------------
def build_features_for_next_date(cur_series: pd.Series, next_date: pd.Timestamp, features_list):
    feat = {}
    feat['dow'] = next_date.dayofweek
    feat['month'] = next_date.month
    feat['day'] = next_date.day
    feat['is_sunday'] = 1 if next_date.weekday() == SUNDAY_WEEKDAY else 0

    for lag in LAGS:
        if len(cur_series) >= lag:
            feat[f'lag_{lag}'] = float(cur_series.iloc[-lag])
        else:
            feat[f'lag_{lag}'] = float(cur_series.mean() if len(cur_series)>0 else 0.0)

    for w in ROLL_WINDOWS:
        if len(cur_series) >= 1:
            window_slice = cur_series.iloc[-w:] if len(cur_series) >= w else cur_series
            feat[f'roll_mean_{w}'] = float(window_slice.mean() if len(window_slice)>0 else cur_series.mean() if len(cur_series)>0 else 0.0)
            feat[f'roll_std_{w}'] = float(window_slice.std() if len(window_slice)>1 else 0.0)
        else:
            feat[f'roll_mean_{w}'] = 0.0
            feat[f'roll_std_{w}'] = 0.0

    return [feat.get(f, 0.0) for f in features_list]

# ------------------------------
# Ensure series has today's date (so forecast starts from today+1)
# ------------------------------
def ensure_series_includes_today(series: pd.Series):
    s = series.copy()
    today = pd.Timestamp(datetime.utcnow().date())  # use UTC date
    if s.index.max() < today:
        # append a row for today using last observed value (or 0 on Sundays)
        last_val = float(s.iloc[-1]) if len(s)>0 else 0.0
        if today.weekday() == SUNDAY_WEEKDAY:
            today_val = 0.0
        else:
            today_val = last_val
        s.loc[today] = today_val
        s = s.sort_index()
    return s

# ------------------------------
# Recursive forecast
# ------------------------------
def recursive_forecast(model, features_list, historical_series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    cur = historical_series.copy()
    # Make sure series index is sorted and includes today
    cur = ensure_series_includes_today(cur)

    out = []
    for i in range(steps):
        next_date = cur.index[-1] + timedelta(days=1)

        # Sundays closed
        if next_date.weekday() == SUNDAY_WEEKDAY:
            yhat = 0.0
        else:
            X_row = build_features_for_next_date(cur, next_date, features_list)
            try:
                yhat = float(model.predict([X_row])[0])
            except Exception:
                yhat = float(cur.iloc[-1]) if len(cur)>0 else 0.0

        yhat = max(0.0, yhat)
        cur.loc[next_date] = yhat
        out.append((next_date, yhat))

    # Convert to dict keyed by YYYY-MM-DD
    forecast_dict = {d.strftime("%Y-%m-%d"): round(float(v), 2) for d,v in out}
    total = sum(v for _,v in out)

    # quick approximate CI +/-12%
    conf_lower = {k: round(v*0.88,2) for k,v in forecast_dict.items()}
    conf_upper = {k: round(v*1.12,2) for k,v in forecast_dict.items()}

    return total, forecast_dict, conf_lower, conf_upper

# ------------------------------
# Fallback simple forecast
# ------------------------------
def fast_fallback_forecast(series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    s = series.copy()
    s = s.sort_index()
    s = s.asfreq('D').fillna(0)
    s[s.index.weekday == SUNDAY_WEEKDAY] = 0.0

    last = float(s.dropna().iloc[-1]) if len(s.dropna())>0 else 0.0
    base = float(s.iloc[-7:].mean()) if len(s) >= 7 else (last if last>0 else float(s.mean() if len(s)>0 else 0.0))

    forecast_vals = np.linspace(base, base*1.03, steps)
    out = []
    cur_last = s.index[-1] if len(s)>0 else pd.Timestamp(datetime.utcnow().date())
    for i in range(steps):
        next_date = (cur_last + timedelta(days=i+1))
        if next_date.weekday() == SUNDAY_WEEKDAY:
            val = 0.0
        else:
            val = float(forecast_vals[i])
        out.append((next_date.strftime("%Y-%m-%d"), round(val,2)))

    forecast_dict = {d:v for d,v in out}
    total = float(sum(v for _,v in out))
    conf_lower = {k: round(v*0.9,2) for k,v in forecast_dict.items()}
    conf_upper = {k: round(v*1.1,2) for k,v in forecast_dict.items()}
    return total, forecast_dict, conf_lower, conf_upper

# ------------------------------
# Startup: load & train (if possible)
# ------------------------------
GLOBAL_MODEL = None
GLOBAL_FEATURES = None
GLOBAL_DAILY = None

def startup_load_and_train():
    global GLOBAL_MODEL, GLOBAL_FEATURES, GLOBAL_DAILY
    print("🔁 Loading data and initializing model...")

    try:
        raw_daily = load_daily_series(REVENUE_FILE)
    except Exception as e:
        print("❌ Failed to load revenue file:", e)
        GLOBAL_DAILY = None
        return

    GLOBAL_DAILY = raw_daily

    # Try to load saved model
    if os.path.exists(MODEL_FILE):
        try:
            saved = joblib.load(MODEL_FILE)
            GLOBAL_MODEL = saved.get('model')
            GLOBAL_FEATURES = saved.get('features')
            print("✅ Loaded saved model from disk.")
            return
        except Exception:
            print("⚠️ Failed to load saved model, will retrain.")

    # Prepare capped, windowed series for training
    try:
        training_series = prepare_series_for_training(raw_daily, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS)
        start = time.time()
        model, features = train_model(training_series)
        elapsed = time.time() - start
        if model is not None:
            GLOBAL_MODEL = model
            GLOBAL_FEATURES = features
            print(f"✅ Trained LightGBM model in {elapsed:.2f}s. Features: {len(features)}")
        else:
            print("⚠️ Not enough data to train LightGBM, will use fallback.")
    except Exception as e:
        print("❌ Training failed:", e)
        traceback.print_exc()
        GLOBAL_MODEL = None
        GLOBAL_FEATURES = None

startup_load_and_train()

# ------------------------------
# Endpoints
# ------------------------------
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
          confidence_intervals: { "lower": {...}, "upper": {...} },
          generated_at: "ISO timestamp"
        }
      }
    """
    start_t = time.time()
    try:
        raw_daily = load_daily_series(REVENUE_FILE)
        if raw_daily is None or len(raw_daily) == 0:
            raise ValueError("No daily revenue data found.")

        # Prepare series for forecast (cap & window to prevent huge outliers & keep recency)
        prepared = prepare_series_for_training(raw_daily, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS)

        # If we have a trained model, use it. Otherwise fallback.
        if GLOBAL_MODEL is not None and GLOBAL_FEATURES is not None:
            total, forecast_dict, conf_lower, conf_upper = recursive_forecast(
                GLOBAL_MODEL, GLOBAL_FEATURES, prepared, steps=FORECAST_STEPS_DEFAULT
            )
        else:
            total, forecast_dict, conf_lower, conf_upper = fast_fallback_forecast(prepared, steps=FORECAST_STEPS_DEFAULT)

        # Post-check: scale down unrealistic totals compared to recent past
        recent_avg_30 = float(prepared.tail(30).mean()) if len(prepared) >= 1 else 0.0
        if recent_avg_30 > 0 and total > recent_avg_30 * 1.5:
            scale = (recent_avg_30 * 1.5) / total
            # scale forecast dict and CI
            for k in list(forecast_dict.keys()):
                forecast_dict[k] = round(forecast_dict[k] * scale, 2)
                conf_lower[k] = round(conf_lower[k] * scale, 2)
                conf_upper[k] = round(conf_upper[k] * scale, 2)
            total = sum(forecast_dict.values())

        elapsed = time.time() - start_t
        print(f"✅ Forecast computed in {elapsed:.2f}s")
        return jsonify({
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": {
                "next_month_total": round(float(total), 2),
                "daily_forecast": forecast_dict,
                "confidence_intervals": {"lower": conf_lower, "upper": conf_upper},
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        })
    except Exception as e:
        print("❌ Forecast failed:")
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
