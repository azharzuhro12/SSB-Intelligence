"""NLP: sentiment (TF-IDF + LogReg, CPU) and offline rule-based Q&A over sales data (no external LLM)."""
import re, json, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from . import data
from .config import MODELS, PROC, REPORTS

def train_sentiment():
    df = data.make_reviews()
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.25, stratify=df.label, random_state=0)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    pipe.fit(Xtr, ytr)
    f1 = float(f1_score(yte, pipe.predict(Xte), average="macro"))
    joblib.dump(pipe, MODELS / "sentiment.joblib")
    (REPORTS / "sentiment_metrics.json").write_text(json.dumps({"macro_f1": f1}))
    return {"macro_f1": f1}

def sentiment(text):
    p = joblib.load(MODELS / "sentiment.joblib")
    prob = float(p.predict_proba([text])[0][1])
    return {"label": "positive" if prob >= .5 else "negative", "score": round(prob, 3)}

def ask(q, df=None):
    """Intent-based Q&A; returns answer plus the executed operation for transparency."""
    df = pd.read_pickle(PROC / "features.pkl") if df is None else df
    ql = q.lower()
    m = re.search(r"(?:toko|store|outlet)\s*(\d+)", ql)
    last = df["Date"].max()
    days = int(re.search(r"(\d+)\s*hari", ql).group(1)) if re.search(r"(\d+)\s*hari", ql) else 30
    d = df[df["Date"] > last - pd.Timedelta(days=days)]
    if m: d = d[d["Store"] == int(m.group(1))]
    scope = f"toko {m.group(1)}" if m else "semua toko"
    if re.search(r"(tertinggi|terbaik|top|paling tinggi)", ql):
        s = d.groupby("Store")["Sales"].sum().nlargest(1)
        return {"answer": f"Toko {s.index[0]} tertinggi: {s.iloc[0]:,.0f} ({days} hari terakhir)", "op": "groupby Store sum -> max"}
    if re.search(r"(terendah|terburuk|paling rendah)", ql):
        s = d[d.Sales > 0].groupby("Store")["Sales"].sum().nsmallest(1)
        return {"answer": f"Toko {s.index[0]} terendah: {s.iloc[0]:,.0f} ({days} hari terakhir)", "op": "groupby Store sum -> min"}
    if re.search(r"(rata|average|avg)", ql):
        return {"answer": f"Rata-rata penjualan harian {scope} ({days} hari): {d[d.Sales > 0].Sales.mean():,.0f}", "op": "mean(Sales>0)"}
    if re.search(r"(total|jumlah)", ql):
        return {"answer": f"Total penjualan {scope} ({days} hari): {d.Sales.sum():,.0f}", "op": "sum(Sales)"}
    if "promo" in ql:
        a = d[(d.Promo == 1) & (d.Sales > 0)].Sales.mean(); b = d[(d.Promo == 0) & (d.Sales > 0)].Sales.mean()
        return {"answer": f"Promo menaikkan rata-rata penjualan {scope} sebesar {(a / b - 1) * 100:.1f}%", "op": "mean promo vs non-promo"}
    return {"answer": "Maaf, pertanyaan belum dikenali. Coba: total/rata-rata/tertinggi/terendah/promo, mis. 'total penjualan toko 3 30 hari'.", "op": None}
