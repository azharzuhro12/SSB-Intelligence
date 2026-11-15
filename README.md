# SSB Demand & Outlet Intelligence

End-to-end AI portfolio project: ETL → demand forecasting → outlet segmentation → NLP (Indonesian) → REST API → drift monitoring & auto-retraining. CPU only. Evaluated on real **Rossmann Store Sales** competition dataset (1,017,209 daily records across 1,115 stores).

See full specification in [PRD.md](PRD.md).

## Quickstart (≤ 15 min)
```bash
pip install -r requirements.txt
python main.py                    # Runs complete pipeline
pytest -q tests                   # 3 test suites incl. p95 latency < 300 ms
python main.py --api              # Starts FastAPI server (port 8050)
python main.py --dashboard        # Starts Streamlit dashboard (port 8501)
```

## Production Evaluation Results (Real Rossmann Dataset)
Holdout evaluation on unseen 28-day future horizon (July 2015):

| Metric | LightGBM Production | Seasonal-Naive Baseline | Target (PRD) | Status |
|---|---|---|---|---|
| **Holdout MAPE** | **9.36%** | 11.96% | ≤ 20.0% | **PASSED** |
| **Holdout WAPE** | **8.94%** | 11.86% | - | **PASSED** |
| **Holdout RMSE** | **859.51** | 1218.02 | - | **PASSED** |
| **API Latency (p95)** | **~75 ms** | - | < 300 ms | **PASSED** |
| **Review Sentiment (F1)** | **0.92** | - | ≥ 0.75 | **PASSED** |
| **Drift Scenarios Detected** | **5 / 5** | - | 5 / 5 | **PASSED** |

### API Usage Example
Header `x-api-key: demo-key` is required (configurable via `SSB_API_KEY`).

```bash
# 1. Forecast next 7 days for Outlet #1
curl -X POST http://localhost:8000/forecast \
  -H 'x-api-key: demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"store": 1, "horizon": 7}'

# 2. Indonesian Customer Review Sentiment Analysis
curl -X POST http://localhost:8000/sentiment \
  -H 'x-api-key: demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "pelayanan ramah dan pengiriman sangat cepat"}'

# 3. Natural Language Data Assistant
curl -X POST http://localhost:8000/ask \
  -H 'x-api-key: demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"question": "toko tertinggi 30 hari"}'
```

## System Architecture & Codebase Layout
- `ssb/data.py`: Multi-source data loader (auto-detects Rossmann raw CSVs or generates synthetic fallback).
- `ssb/etl.py`: Idempotent data pipeline with strict schema & range validation gates.
- `ssb/forecast.py`: LightGBM time-series regressor with rolling-origin CV and automated model registry.
- `ssb/outlet.py`: K-Means clustering (5 operational tiers across 1,115 stores, silhouette evaluated).
- `ssb/nlp.py`: TF-IDF n-gram sentiment engine + offline intent-based business Q&A engine.
- `ssb/monitor.py`: Population Stability Index (PSI) & Kolmogorov-Smirnov drift detection with automated retrain trigger.
- `ssb/api.py`: FastAPI endpoints with Pydantic validation & timing benchmarks.
- `ssb/dashboard.py`: Streamlit visual explorer for outlet sales history & forecast curves.
- `docs/`: Complete [MODEL_CARD.md](docs/MODEL_CARD.md), [PIPELINE.md](docs/PIPELINE.md), and [USER_GUIDE.md](docs/USER_GUIDE.md).
