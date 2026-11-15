from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
HORIZON = 28
for p in (RAW, PROC, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
