"""Outlet segmentation (KMeans on sales profile)."""
import pandas as pd, json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from .config import PROC, REPORTS

def segment():
    df = pd.read_pickle(PROC / "features.pkl")
    o = df[df["Sales"] > 0]
    last = o["Date"].max()
    f = o.groupby("Store").agg(avg_sales=("Sales", "mean"), cv=("Sales", lambda s: s.std() / s.mean()),
                               promo_share=("Promo", "mean"))
    recent = o[o["Date"] > last - pd.Timedelta(days=90)].groupby("Store")["Sales"].mean()
    prev = o[(o["Date"] <= last - pd.Timedelta(days=90)) & (o["Date"] > last - pd.Timedelta(days=180))].groupby("Store")["Sales"].mean()
    f["growth_90d"] = (recent / prev - 1).fillna(0)
    X = StandardScaler().fit_transform(f)
    best = max(range(3, 7), key=lambda k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)))
    km = KMeans(best, n_init=10, random_state=0).fit(X)
    f["segment"] = km.labels_
    rank = f.groupby("segment")["avg_sales"].mean().rank(ascending=False).astype(int)
    names = {1: "Top performer", 2: "Strong", 3: "Mid", 4: "Developing", 5: "Low", 6: "Lowest"}
    f["segment_label"] = f["segment"].map(rank).map(names)
    f.to_csv(REPORTS / "outlet_segments.csv")
    out = {"k": best, "silhouette": float(silhouette_score(X, km.labels_)), "sizes": f["segment_label"].value_counts().to_dict()}
    (REPORTS / "outlet_summary.json").write_text(json.dumps(out, indent=2))
    return out
