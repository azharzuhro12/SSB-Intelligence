"""Streamlit dashboard. Run: streamlit run ssb/dashboard.py"""
import pandas as pd, streamlit as st
from ssb import forecast
from ssb.config import PROC, REPORTS
st.title("SSB Demand & Outlet Intelligence")
df = pd.read_pickle(PROC / "features.pkl")
store = st.selectbox("Store", sorted(df.Store.unique()))
h = df[df.Store == store].set_index("Date")["Sales"].tail(120)
f = pd.DataFrame(forecast.forecast_store(store, 28, df)).set_index("date")["forecast"]
f.index = pd.to_datetime(f.index)
st.line_chart(pd.concat([h.rename("actual"), f.rename("forecast")], axis=1))
st.subheader("Outlet segments")
st.dataframe(pd.read_csv(REPORTS / "outlet_segments.csv"))
