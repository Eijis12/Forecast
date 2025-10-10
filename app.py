# app.py (updated)
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

# optional timezone support (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
    MANILA_TZ = ZoneInfo("Asia/Manila")
except Exception:
    MANILA_TZ = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(__file__)
REVENUE_FILE = os.path.join(BASE_DIR, "DentalRecords_RevenueForecasting.xlsx")
MODEL_FILE = os.path.join(BASE_DIR, "lgb_revenue.pkl")
FORECASTS_DIR = os.path.join(BASE_DIR, "forecasts")
os.makedirs(FORECASTS_DIR, exist_ok=True)

LAGS = [1, 2, 3, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 30]
FORECAST_STEPS_DEFAULT = 30
OUTLIER_QUANTILE = 0.95   # clipping extreme days
TRAIN_WINDOW_DAYS = 365   # use last 12 months
SUNDAY_WEEKDAY = 6        # Monday=0 ... Sunday=6
TARGET_MIN = 50000
TARGET_MAX = 100000

# ---------------- Utilities ----------------
def now_local_date():
    """Return today's date in Asia/Manila if available, else UTC date."""
    if MANILA_TZ:
        return datetime.now(MANILA_TZ).date()
    return datetime.utcnow().date()

def load_daily_series(file_path=REVENUE_FILE):
    df = pd.read_excel(file_path)
    amount_cols = [c for c in df.columns if 'amount' in c.lower()]
    if not amount_cols:
        raise ValueError("No column with 'amount' found in dataset.")
    amount_col = amount_cols[0]

    if 'DATE' not in df.columns:
        if all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-' + df['DAY'].astype(str))
        elif all(col in df.columns for col in ['YEAR', 'MONTH']):
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        else:
            raise ValueError("No DATE or YEAR/MONTH(DAY) columns found in dataset.")

    df = df.dropna(subset=['DATE']).copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    daily = df.groupby('DATE')[amount_col].sum().sort_index()
    daily = daily.asfreq('D').fillna(0)
    daily.name = 'y'
    return daily

def prepare_series_for_training(series: pd.Series, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS):
    s = series.copy().sort_index()
    if len(s) > days_window:
        s = s.iloc[-days_window:]
    cap = float(s.quantile(cap_quantile))
    s = s.clip(upper=cap)
    # explicitly set Sundays to 0 in training data (clinic closed)
    s[s.index.weekday == SUNDAY_WEEKDAY] = 0.0
    s.name = 'y'
    return s

# ---------------- Feature engineering ----------------
def build_feature_table(series: pd.Series):
    df = pd.DataFrame(series).rename(columns={series.name: 'y'})
    df = df.sort_index()
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['is_sunday'] = (df.index.dayofweek == SUNDAY_WEEKDAY).astype(int)

    for lag in LAGS:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    for w in ROLL_WINDOWS:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].shift(1).rolling(w).std().fillna(0)

    df = df.dropna()
    features = [c for c in df.columns if c != 'y']
    return df, features

# ---------------- Training ----------------
def train_model(series: pd.Series):
    if len(series) < 14:
        return None, None

    df_feat, features = build_feature_table(series)
    X = df_feat[features]
    y = df_feat['y']

    val_days = min(30, int(len(X) * 0.2))
    if val_days < 1:
        split_idx = len(X)
    else:
        split_idx = len(X) - val_days

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain) if len(X_val) > 0 else None

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'feature_pre_filter': False
    }

    if dval is not None:
        booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=False)
    else:
        booster = lgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)

    try:
        joblib.dump({'model': booster, 'features': features}, MODEL_FILE)
    except Exception:
        pass

    return booster, features

# ---------------- Forecast helpers ----------------
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
            feat[f'lag_{lag}'] = float(cur_series.mean() if len(cur_series) > 0 else 0.0)

    for w in ROLL_WINDOWS:
        if len(cur_series) >= 1:
            window_slice = cur_series.iloc[-w:] if len(cur_series) >= w else cur_series
            feat[f'roll_mean_{w}'] = float(window_slice.mean() if len(window_slice) > 0 else cur_series.mean() if len(cur_series) > 0 else 0.0)
            feat[f'roll_std_{w}'] = float(window_slice.std() if len(window_slice) > 1 else 0.0)
        else:
            feat[f'roll_mean_{w}'] = 0.0
            feat[f'roll_std_{w}'] = 0.0

    return [feat.get(f, 0.0) for f in features_list]

def compute_weekday_multipliers(series: pd.Series):
    # average revenue by day-of-week (0=Mon..6=Sun)
    try:
        dow_mean = series.groupby(series.index.dayofweek).mean()
        overall = dow_mean.mean() if not dow_mean.empty else 0.0
        if overall <= 0:
            # fallback: typical dental pattern (Mon slightly lower, Tue-Fri higher, Sat moderate, Sun zero)
            base = {0:0.9, 1:1.05, 2:1.1, 3:1.0, 4:1.05, 5:0.95, 6:0.0}
            return {k: float(v) for k,v in base.items()}
        multipliers = (dow_mean / overall).to_dict()
        # ensure Sunday multiplier is zero
        multipliers[SUNDAY_WEEKDAY] = 0.0
        # if any missing weekdays, fill with 1.0 except Sunday
        for d in range(7):
            if d not in multipliers:
                multipliers[d] = 0.0 if d == SUNDAY_WEEKDAY else 1.0
        return {int(k): float(v) for k,v in multipliers.items()}
    except Exception:
        return {0:0.9,1:1.05,2:1.1,3:1.0,4:1.05,5:0.95,6:0.0}

def recursive_forecast_with_multipliers(model, features_list, historical_series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    cur = historical_series.copy().sort_index()
    # ensure series includes today; if not, append today with last observed value (or 0 if Sunday)
    today = now_local_date()
    today_ts = pd.Timestamp(today)
    if cur.index.max() < today_ts:
        last_val = float(cur.iloc[-1]) if len(cur) > 0 else 0.0
        cur.loc[today_ts] = 0.0 if today_ts.weekday() == SUNDAY_WEEKDAY else last_val
        cur = cur.sort_index()

    multipliers = compute_weekday_multipliers(historical_series)
    out = []
    for _ in range(steps):
        next_date = cur.index[-1] + timedelta(days=1)
        dow = int(next_date.weekday())
        if dow == SUNDAY_WEEKDAY:
            yhat = 0.0
        else:
            X_row = build_features_for_next_date(cur, next_date, features_list)
            try:
                yhat = float(model.predict([X_row])[0])
            except Exception:
                yhat = float(cur.iloc[-1]) if len(cur) > 0 else 0.0

            # apply weekday multiplier to add realistic day-of-week variation
            multiplier = multipliers.get(dow, 1.0)
            yhat = float(yhat * multiplier)

        yhat = max(0.0, yhat)
        cur.loc[next_date] = yhat
        out.append((next_date, yhat))

    forecast_dict = {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in out}
    total = sum(v for _, v in out)
    conf_lower = {k: round(v * 0.88, 2) for k, v in forecast_dict.items()}
    conf_upper = {k: round(v * 1.12, 2) for k, v in forecast_dict.items()}
    return total, forecast_dict, conf_lower, conf_upper

def fast_fallback_forecast(series: pd.Series, steps=FORECAST_STEPS_DEFAULT):
    s = series.copy().sort_index().asfreq('D').fillna(0)
    s[s.index.weekday == SUNDAY_WEEKDAY] = 0.0
    base = float(s.iloc[-7:].mean()) if len(s) >= 7 else float(s.mean() if len(s) > 0 else 0.0)
    forecast_vals = np.linspace(base, base * 1.03, steps)
    out = []
    cur_last = s.index[-1] if len(s) > 0 else pd.Timestamp(now_local_date())
    for i in range(steps):
        next_date = (cur_last + timedelta(days=i + 1))
        if next_date.weekday() == SUNDAY_WEEKDAY:
            val = 0.0
        else:
            val = float(forecast_vals[i])
        out.append((next_date.strftime("%Y-%m-%d"), round(val, 2)))
    forecast_dict = dict(out)
    total = float(sum(v for _, v in out))
    conf_lower = {k: round(v * 0.9, 2) for k, v in forecast_dict.items()}
    conf_upper = {k: round(v * 1.1, 2) for k, v in forecast_dict.items()}
    return total, forecast_dict, conf_lower, conf_upper

# ---------------- Save forecast ----------------
def save_forecast_to_csv(forecast_dict, generated_at_iso, total):
    try:
        ts = generated_at_iso.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        filename = f"forecast_{ts}.csv"
        path = os.path.join(FORECASTS_DIR, filename)
        df_out = pd.DataFrame(list(forecast_dict.items()), columns=['date', 'forecast'])
        df_out['generated_at'] = generated_at_iso
        df_out['total_forecast'] = round(total, 2)
        df_out.to_csv(path, index=False)
        return path
    except Exception as e:
        print("Failed to save forecast:", e)
        return None

# ---------------- Startup: load & train ----------------
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

    if os.path.exists(MODEL_FILE):
        try:
            cached = joblib.load(MODEL_FILE)
            GLOBAL_MODEL = cached.get('model')
            GLOBAL_FEATURES = cached.get('features')
            print("✅ Loaded saved model from disk.")
        except Exception:
            print("⚠️ Saved model load failed — retraining.")

    if GLOBAL_MODEL is None:
        try:
            training_series = prepare_series_for_training(raw_daily, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS)
            start = time.time()
            model, features = train_model(training_series)
            duration = time.time() - start
            if model is not None:
                GLOBAL_MODEL = model
                GLOBAL_FEATURES = features
                print(f"✅ Trained model in {duration:.2f}s (features={len(features)})")
            else:
                print("⚠️ Not enough data to train; will use fallback.")
        except Exception as e:
            print("❌ Training failed:")
            traceback.print_exc()

startup_load_and_train()

# ---------------- Routes ----------------
@app.route("/")
def home():
    return jsonify({"message": "Revenue forecasting service is UP"})

@app.route("/api/revenue/forecast", methods=["GET"])
def revenue_forecast():
    start_t = time.time()
    save_flag = request.args.get('save', "false").lower() in ("1", "true", "yes")
    try:
        raw_daily = load_daily_series(REVENUE_FILE)
        if raw_daily is None or len(raw_daily) == 0:
            raise ValueError("No daily revenue data found.")

        prepared = prepare_series_for_training(raw_daily, cap_quantile=OUTLIER_QUANTILE, days_window=TRAIN_WINDOW_DAYS)

        if GLOBAL_MODEL is not None and GLOBAL_FEATURES is not None:
            total, forecast_dict, conf_lower, conf_upper = recursive_forecast_with_multipliers(
                GLOBAL_MODEL, GLOBAL_FEATURES, prepared, steps=FORECAST_STEPS_DEFAULT
            )
        else:
            total, forecast_dict, conf_lower, conf_upper = fast_fallback_forecast(prepared, steps=FORECAST_STEPS_DEFAULT)

        # The forecast is already generated starting from today+1 because recursive uses local today when needed.
        # Ensure dates start at tomorrow in case older index behavior produced different labels:
        today = now_local_date()
        start_date = pd.Timestamp(today + timedelta(days=1))
        # Re-label forecast days sequentially starting from start_date to ensure alignment
        ordered_vals = list(forecast_dict.values())
        rebased = {}
        rebased_lower = {}
        rebased_upper = {}
        for i, v in enumerate(ordered_vals):
            d = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            rebased[d] = round(float(v), 2)
            # for CI, if lengths mismatch use the value or compute simple +/-10%
            old_keys = list(forecast_dict.keys())
            try:
                oldk = old_keys[i]
                rebased_lower[d] = round(float(conf_lower.get(oldk, rebased[d]*0.9)), 2)
                rebased_upper[d] = round(float(conf_upper.get(oldk, rebased[d]*1.1)), 2)
            except Exception:
                rebased_lower[d] = round(rebased[d]*0.9, 2)
                rebased_upper[d] = round(rebased[d]*1.1, 2)

        forecast_dict = rebased
        conf_lower = rebased_lower
        conf_upper = rebased_upper

        # Scale to realistic total range (50k - 100k)
        total = sum(forecast_dict.values())
        target = (TARGET_MIN + TARGET_MAX) / 2.0
        if total <= 0:
            scale = 1.0
        elif total < TARGET_MIN or total > TARGET_MAX:
            scale = target / total
        else:
            scale = 1.0

        if scale != 1.0:
            for k in list(forecast_dict.keys()):
                forecast_dict[k] = round(forecast_dict[k] * scale, 2)
                conf_lower[k] = round(conf_lower[k] * scale, 2)
                conf_upper[k] = round(conf_upper[k] * scale, 2)
            total = sum(forecast_dict.values())

        generated_at = datetime.utcnow().isoformat() + "Z"

        saved_path = None
        if save_flag:
            saved_path = save_forecast_to_csv(forecast_dict, generated_at, total)

        elapsed = time.time() - start_t
        print(f"✅ Forecast computed in {elapsed:.2f}s (total ₱{total:,.2f}) saved={bool(saved_path)}")

        resp = {
            "status": "success",
            "message": "Revenue forecast generated successfully",
            "data": {
                "next_month_total": round(float(total), 2),
                "daily_forecast": forecast_dict,
                "confidence_intervals": {"lower": conf_lower, "upper": conf_upper},
                "generated_at": generated_at
            }
        }
        if saved_path:
            resp['data']['saved_path'] = saved_path

        return jsonify(resp)
    except Exception as e:
        print("❌ Forecast failed:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
