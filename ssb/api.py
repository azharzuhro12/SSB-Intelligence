"""FastAPI service. Run: uvicorn ssb.api:app --reload"""
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import os
from . import forecast, nlp
from .config import PROC

app = FastAPI(title="SSB Demand & Outlet Intelligence", version="1.0")
API_KEY = os.getenv("SSB_API_KEY", "demo-key")
_df = None
def df():
    global _df
    if _df is None: _df = pd.read_pickle(PROC / "features.pkl")
    return _df
def auth(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY: raise HTTPException(401, "invalid API key")

class FReq(BaseModel): store: int; horizon: int = 28
class SReq(BaseModel): text: str
class QReq(BaseModel): question: str

@app.get("/health")
def health(): return {"status": "ok"}
@app.post("/forecast", dependencies=[Depends(auth)])
def fc(r: FReq):
    if not 1 <= r.horizon <= 28: raise HTTPException(422, "horizon must be 1..28")
    try: return {"store": r.store, "forecast": forecast.forecast_store(r.store, r.horizon, df())}
    except KeyError: raise HTTPException(404, "store not found")
@app.post("/sentiment", dependencies=[Depends(auth)])
def sent(r: SReq): return nlp.sentiment(r.text)
@app.post("/ask", dependencies=[Depends(auth)])
def ask(r: QReq): return nlp.ask(r.question, df())
