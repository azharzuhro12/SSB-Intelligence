"""SSB Demand & Outlet Intelligence — Main Entrypoint.
Usage:
  python main.py             # Runs complete end-to-end ML pipeline
  python main.py --pipeline  # Runs pipeline
  python main.py --api       # Starts FastAPI server on port 8050
  python main.py --dashboard # Starts Streamlit dashboard on port 8501
"""

import sys
import json
import os

def run_pipeline():
    from ssb import etl, forecast, outlet, nlp, monitor
    print("[1/5] Running ETL Pipeline...")
    res_etl = etl.run()
    print("      ETL Completed:", res_etl)
    
    print("[2/5] Training LightGBM Demand Forecasting Model...")
    r = forecast.train()
    print("      Holdout Performance:", json.dumps(r["holdout"], indent=2))
    print(f"      Model Version: v{r.get('version')} (Promoted: {r.get('promoted')})")
    
    print("[3/5] Performing Outlet Segmentation (K-Means)...")
    res_outlet = outlet.segment()
    print("      Outlet Clusters:", res_outlet)
    
    print("[4/5] Training NLP Sentiment Analysis Model...")
    res_nlp = nlp.train_sentiment()
    print("      NLP F1-Score:", res_nlp)
    
    print("[5/5] Executing Drift Regression Scenarios & Health Audit...")
    scen = monitor.simulate_scenarios()
    audit = monitor.run(retrain=False)
    print("      Drift Scenarios Detected:", scen)
    print("      System Drift Status:", "STABLE" if not audit.get("drift") else "DRIFT DETECTED")
    print("\n✓ All pipeline stages executed successfully!")

if __name__ == "__main__":
    if "--api" in sys.argv:
        import uvicorn
        port = int(os.getenv("PORT", 8050))
        print(f"Starting FastAPI on port {port}...")
        uvicorn.run("ssb.api:app", host="0.0.0.0", port=port, reload=True)
    elif "--dashboard" in sys.argv:
        import subprocess
        print("Starting Streamlit Dashboard on port 8501...")
        subprocess.run(["streamlit", "run", "ssb/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
    else:
        run_pipeline()
