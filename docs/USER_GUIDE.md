# User Guide
1. `python main.py` to build data and models (or `python main.py --dashboard` to open the UI directly).
2. Start API; send `x-api-key: demo-key` (override with env `SSB_API_KEY`).
3. `/forecast` {store, horizon 1..28}; `/sentiment` {text}; `/ask` {question}.
4. Q&A understands: total, rata-rata, tertinggi, terendah, promo; optional `toko N` and `N hari`.
5. Dashboard: pick a store to see 120 days history + 28-day forecast and outlet segments.
6. Real data: place Kaggle Rossmann `train.csv` and `store.csv` in `data/raw/` and rerun.
