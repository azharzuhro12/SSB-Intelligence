"""Drift detection (PSI + KS) and automatic retraining."""
import json, time, numpy as np, pandas as pd
from scipy.stats import ks_2samp
from . import forecast
from .config import PROC, REPORTS

def psi(a, b, bins=10):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))

def check(ref, cur, cols=("Sales", "roll_mean_7")):
    out = {}
    for c in cols:
        a, b = ref[c].dropna().values, cur[c].dropna().values
        out[c] = {"psi": psi(a, b), "ks_p": float(ks_2samp(a, b).pvalue)}
    out["drift"] = any(v["psi"] > 0.25 for v in out.values() if isinstance(v, dict))
    return out

def _reference(o, last, window):
    """Same calendar window one year earlier (controls for seasonality); fallback: preceding windows."""
    yago = last - pd.Timedelta(days=365)
    ref = o[(o.Date > yago - pd.Timedelta(days=window)) & (o.Date <= yago)]
    if len(ref) < 100:
        ref = o[(o.Date <= last - pd.Timedelta(days=window)) & (o.Date > last - pd.Timedelta(days=window * 4))]
    return ref

def run(df=None, window=60, retrain=True):
    df = pd.read_pickle(PROC / "features.pkl") if df is None else df
    o = df[df.Sales > 0]
    last = o.Date.max()
    cur = o[o.Date > last - pd.Timedelta(days=window)]
    ref = _reference(o, last, window)
    res = check(ref, cur)
    res["retrained"] = None
    if res["drift"] and retrain:
        res["retrained"] = forecast.train(df.dropna(subset=forecast.FEATURES))
    res["ts"] = time.time()
    (REPORTS / "drift_report.json").write_text(json.dumps(res, indent=2, default=str))
    return res

def simulate_scenarios():
    """Regression test: 5 synthetic drifts must all be detected; a clean sample must not."""
    df = pd.read_pickle(PROC / "features.pkl"); o = df[df.Sales > 0]
    last = o.Date.max(); win = 60
    ref = _reference(o, last, win)
    cur = o[o.Date > last - pd.Timedelta(days=win)].copy()
    scen = {"demand_drop_40%": 0.6, "demand_spike_60%": 1.6, "demand_drop_25%": 0.75, "spike_35%": 1.35, "crash_55%": 0.45}
    hits = {}
    for n, f in scen.items():
        c = cur.copy()
        for col in ("Sales", "roll_mean_7"): c[col] = c[col] * f
        hits[n] = check(ref, c)["drift"]
    hits["clean_no_false_alarm"] = not check(ref, cur.sample(frac=.5, random_state=0))["drift"]
    (REPORTS / "drift_scenarios.json").write_text(json.dumps(hits, indent=2))
    return hits
