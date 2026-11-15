"""ETL: extract -> validate -> clean -> features. Idempotent (overwrites outputs)."""
import pandas as pd, numpy as np
from . import data
from .config import PROC

class DataQualityError(Exception): ...

def validate(df):
    checks = {
        "no_null_keys": df[["Store", "Date"]].notna().all().all(),
        "no_duplicates": not df.duplicated(["Store", "Date"]).any(),
        "sales_nonneg": (df["Sales"].dropna() >= 0).all(),
        "promo_binary": df["Promo"].isin([0, 1]).all(),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad: raise DataQualityError(f"Failed checks: {bad}")
    return checks

def add_features(df):
    df = df.sort_values(["Store", "Date"]).copy()
    d = df["Date"]
    df["dow"], df["month"], df["day"], df["week"] = d.dt.dayofweek, d.dt.month, d.dt.day, d.dt.isocalendar().week.astype(int)
    g = df.groupby("Store")["Sales"]
    for lag in (28, 35, 56):  # lags >= horizon => no leakage in 28-day recursive-free forecast
        df[f"lag_{lag}"] = g.shift(lag)
    for w in (7, 28):
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(28).rolling(w).mean())
    return df

def run():
    df, stores, source = data.load()
    df = df.drop_duplicates(["Store", "Date"])
    checks = validate(df)
    df = df.merge(stores, on="Store", how="left")
    for c in ("StoreType", "Assortment"):
        df[c] = df[c].astype("category").cat.codes
    feats = add_features(df)
    feats.to_pickle(PROC / "features.pkl")
    stores.to_pickle(PROC / "stores.pkl")
    return {"source": source, "rows": len(feats), "checks": checks}

if __name__ == "__main__":
    print(run())
