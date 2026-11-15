"""Forecast model: LightGBM vs seasonal-naive baseline, rolling-origin CV, holdout eval."""
import json, time, joblib, numpy as np, pandas as pd
import lightgbm as lgb
from .config import PROC, MODELS, REPORTS, HORIZON

FEATURES = ["Store", "Promo", "StateHoliday", "StoreType", "Assortment", "CompetitionDistance",
            "dow", "month", "day", "week", "lag_28", "lag_35", "lag_56", "roll_mean_7", "roll_mean_28"]

def mape(y, p):  # on non-zero actuals only
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)
def wape(y, p): return float(np.abs(y - p).sum() / max(y.sum(), 1) * 100)
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def metrics(y, p): return {"MAPE": mape(y, p), "WAPE": wape(y, p), "RMSE": rmse(y, p)}

def _fit(tr):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                          colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    m.fit(tr[FEATURES], tr["Sales"])
    return m

def train(df=None, register=True):
    df = pd.read_pickle(PROC / "features.pkl") if df is None else df
    df = df.dropna(subset=FEATURES)
    open_df = df[df["Sales"] > 0]
    end = df["Date"].max()
    cv = []
    for k in (3, 2, 1):  # rolling-origin folds
        cut = end - pd.Timedelta(days=HORIZON * (k + 1))
        tr = open_df[open_df["Date"] <= cut]
        va = df[(df["Date"] > cut) & (df["Date"] <= cut + pd.Timedelta(days=HORIZON))]
        va = va[va["Sales"] > 0]
        cv.append(metrics(va["Sales"].values, _fit(tr).predict(va[FEATURES])))
    cut = end - pd.Timedelta(days=HORIZON)
    tr, te = open_df[open_df["Date"] <= cut], df[(df["Date"] > cut) & (df["Sales"] > 0)]
    model = _fit(tr)
    hold = metrics(te["Sales"].values, model.predict(te[FEATURES]))
    base = metrics(te["Sales"].values, te["lag_28"].values)  # seasonal-naive (same weekday 4w ago)
    res = {"cv": cv, "holdout": hold, "baseline_seasonal_naive": base, "cutoff": str(cut.date())}
    if register:
        full = _fit(open_df)
        res = register_model(full, res)
    return res

def register_model(model, res):
    """Tiny file-based registry. Promote only if better than current (lower holdout MAPE)."""
    reg_path = MODELS / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"versions": [], "current": None}
    v = len(reg["versions"]) + 1
    cur = next((x for x in reg["versions"] if x["version"] == reg["current"]), None)
    promoted = cur is None or res["holdout"]["MAPE"] <= cur["holdout"]["MAPE"]
    joblib.dump(model, MODELS / f"forecast_v{v}.joblib")
    reg["versions"].append({"version": v, "ts": time.time(), "holdout": res["holdout"], "cv": res["cv"]})
    if promoted: reg["current"] = v
    reg_path.write_text(json.dumps(reg, indent=2))
    res.update(version=v, promoted=promoted)
    (REPORTS / "forecast_metrics.json").write_text(json.dumps(res, indent=2))
    return res

def load_current():
    reg = json.loads((MODELS / "registry.json").read_text())
    return joblib.load(MODELS / f"forecast_v{reg['current']}.joblib")

def forecast_store(store, horizon=HORIZON, df=None, promo_days=None):
    """Forecast next `horizon` days (<=28) for a store using lag>=28 features (no recursion needed)."""
    df = pd.read_pickle(PROC / "features.pkl") if df is None else df
    hist = df[df["Store"] == store].sort_values("Date")
    if hist.empty: raise KeyError(store)
    last = hist["Date"].max()
    fut = pd.DataFrame({"Date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon)})
    sales = hist.set_index("Date")["Sales"]
    static = hist.iloc[-1]
    ext = sales.reindex(pd.date_range(sales.index.min(), fut["Date"].max()))
    rows = []
    promo_set = set(promo_days) if promo_days is not None else set()
    for idx, d in enumerate(fut["Date"]):
        is_promo = 1 if (idx in promo_set or d.strftime("%Y-%m-%d") in promo_set) else 0
        r = {"Store": store, "Promo": is_promo, "StateHoliday": 0, "StoreType": static["StoreType"],
             "Assortment": static["Assortment"], "CompetitionDistance": static["CompetitionDistance"],
             "dow": d.dayofweek, "month": d.month, "day": d.day, "week": int(d.isocalendar()[1])}
        for lag in (28, 35, 56): r[f"lag_{lag}"] = ext.get(d - pd.Timedelta(days=lag), np.nan)
        for w in (7, 28):
            end = d - pd.Timedelta(days=28)
            r[f"roll_mean_{w}"] = ext.loc[end - pd.Timedelta(days=w - 1):end].mean()
        rows.append(r)
    X = pd.DataFrame(rows)
    pred = np.clip(load_current().predict(X[FEATURES]), 0, None)
    closed = {d for d in range(7) if (hist[hist.dow == d]["Sales"] == 0).mean() > 0.9}
    pred[np.isin(fut["Date"].dt.dayofweek.values, list(closed))] = 0
    return [{"date": str(d.date()), "forecast": round(float(p), 1)} for d, p in zip(fut["Date"], pred)]
