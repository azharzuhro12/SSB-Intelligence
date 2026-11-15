import time, pandas as pd, pytest
from fastapi.testclient import TestClient
from ssb import etl, nlp
from ssb.api import app
H = {"x-api-key": "demo-key"}
def test_validate_rejects_duplicates():
    df = pd.DataFrame({"Store": [1, 1], "Date": ["2020-01-01"] * 2, "Sales": [1, 1], "Promo": [0, 0]})
    with pytest.raises(etl.DataQualityError): etl.validate(df)
def test_api():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.post("/forecast", json={"store": 1}).status_code == 401
    r = c.post("/forecast", json={"store": 1, "horizon": 7}, headers=H); assert len(r.json()["forecast"]) == 7
    assert c.post("/sentiment", json={"text": "pelayanan ramah dan cepat"}, headers=H).json()["label"] == "positive"
    t = []
    for _ in range(30):
        s = time.perf_counter(); c.post("/forecast", json={"store": 2}, headers=H); t.append(time.perf_counter() - s)
    assert sorted(t)[int(.95 * len(t)) - 1] < 0.3
def test_qa():
    assert "Total" in nlp.ask("total penjualan toko 3 30 hari")["answer"]
