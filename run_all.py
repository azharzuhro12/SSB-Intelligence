"""One-shot run: ETL -> forecast -> outlet -> NLP -> drift. Usage: python run_all.py"""
import json
from ssb import etl, forecast, outlet, nlp, monitor
print("ETL     :", etl.run())
r = forecast.train(); print("Forecast:", json.dumps({k: r[k] for k in ("holdout", "baseline_seasonal_naive", "version", "promoted")}, indent=1))
print("Outlet  :", outlet.segment())
print("NLP     :", nlp.train_sentiment())
print("Drift   :", monitor.simulate_scenarios())
print("Monitor :", {k: v for k, v in monitor.run(retrain=False).items() if k in ("drift",)})
