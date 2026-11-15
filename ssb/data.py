"""Data source. Uses real Rossmann files (data/raw/train.csv + store.csv) if present,
otherwise generates a synthetic Rossmann-schema dataset (Kaggle needs credentials)."""
import numpy as np, pandas as pd
from .config import RAW

def generate_synthetic(n_stores=40, days=730, seed=42, end="2026-10-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days)
    stores = pd.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "StoreType": rng.choice(list("abcd"), n_stores),
        "Assortment": rng.choice(list("abc"), n_stores),
        "CompetitionDistance": rng.integers(100, 20000, n_stores),
    })
    base = rng.uniform(3000, 9000, n_stores)
    rows = []
    for i, s in stores.iterrows():
        dow = np.array([1.1, 1.0, 0.95, 0.95, 1.05, 1.2, 0.0])[np.asarray(dates.dayofweek)]
        season = 1 + 0.15 * np.sin(2 * np.pi * np.asarray(dates.dayofyear) / 365.25)
        trend = 1 + 0.0003 * np.arange(days)
        promo = (rng.random(days) < 0.25).astype(int)
        holiday = (rng.random(days) < 0.03).astype(int)
        sales = base[i] * dow * season * trend * (1 + 0.25 * promo) * (1 - 0.3 * holiday)
        sales = sales * rng.normal(1, 0.07, days)
        sales[dow == 0] = 0
        rows.append(pd.DataFrame({"Store": s.Store, "Date": dates, "Sales": sales.round(0),
                                  "Promo": promo, "StateHoliday": holiday}))
    return pd.concat(rows, ignore_index=True), stores

def load():
    tr, st = RAW / "train.csv", RAW / "store.csv"
    if tr.exists() and st.exists():
        df = pd.read_csv(tr, parse_dates=["Date"], low_memory=False)
        df["StateHoliday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
        return df[["Store", "Date", "Sales", "Promo", "StateHoliday"]], pd.read_csv(st)[
            ["Store", "StoreType", "Assortment", "CompetitionDistance"]].fillna({"CompetitionDistance": 0}), "rossmann"
    df, stores = generate_synthetic()
    return df, stores, "synthetic"

REVIEWS = {
 1: ["produk selalu segar dan lengkap", "pelayanan ramah dan cepat", "harga murah dan promo banyak", "toko bersih dan rapi", "barang berkualitas sangat puas", "kasir cepat, recommended", "stok selalu tersedia mantap"],
 0: ["barang sering kosong", "pelayanan lambat dan kurang ramah", "harga mahal tidak sesuai", "toko kotor dan berantakan", "produk kadaluarsa kecewa", "antrian kasir lama sekali", "kemasan rusak buruk"],
}
def make_reviews(n=600, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = int(rng.random() < 0.55)
        t = " ".join(rng.choice(REVIEWS[y], size=rng.integers(1, 3), replace=False))
        if rng.random() < 0.08: y, t = 1 - y, t  # label noise
        out.append((t, y))
    return pd.DataFrame(out, columns=["text", "label"])
