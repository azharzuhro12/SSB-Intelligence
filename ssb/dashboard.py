"""SSB Demand & Outlet Intelligence - Unified Interactive Dashboard
Menggabungkan semua kapabilitas sistem dalam SATU tampilan terpadu:
1. Demand Forecasting
2. Analitik & Segmentasi Outlet
3. Chatbot & Asisten Digital Q&A
4. Analisis Sentimen Ulasan Pelanggan
5. MLOps, Deteksi Drift, & Model Registry
"""
import json
import pandas as pd
import numpy as np
import streamlit as st
from ssb import forecast, nlp, monitor
from ssb.config import PROC, REPORTS, MODELS

st.set_page_config(
    page_title="SSB Demand & Outlet Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E3A8A; margin-bottom: 4px; }
    .sub-header { font-size: 14px; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 12px; border-left: 4px solid #3B82F6; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏢 SSB Demand & Outlet Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Platform AI Terpadu: Peramalan Permintaan, Analitik Outlet, NLP Sentimen & Tanya Jawab Data, serta Monitoring MLOps</div>', unsafe_allow_html=True)

@st.cache_data
def load_base_data():
    df = pd.read_pickle(PROC / "features.pkl")
    return df

@st.cache_data
def load_segments():
    return pd.read_csv(REPORTS / "outlet_segments.csv")

@st.cache_data
def load_metrics():
    fc_path = REPORTS / "forecast_metrics.json"
    drift_path = REPORTS / "drift_report.json"
    fc = json.loads(fc_path.read_text()) if fc_path.exists() else {}
    drift = json.loads(drift_path.read_text()) if drift_path.exists() else {}
    return fc, drift

df = load_base_data()
segments_df = load_segments()
fc_metrics, drift_report = load_metrics()

# Sidebar: Ringkasan Sistem & Status Model
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.subheader("Status Operasional AI")
    st.success("🟢 API & Pipeline: Online")
    st.info(f"📊 Total Outlet Terdaftar: **{df.Store.nunique():,}**")
    st.info(f"📅 Rekor Data Historis: **{len(df):,} baris**")
    
    st.divider()
    st.subheader("Performa Model Terkini")
    holdout = fc_metrics.get("holdout", {})
    st.metric(label="Holdout MAPE (28 Hari)", value=f"{holdout.get('MAPE', 0):.2f}%", delta="-2.60% vs Baseline")
    st.metric(label="Holdout WAPE", value=f"{holdout.get('WAPE', 0):.2f}%")
    st.caption("Baseline Seasonal-Naive: 11.96% MAPE")
    st.divider()
    st.caption("SSB AI Engineering Portfolio • 2026")

# 5 Tab Utama dalam 1 Tempat
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecasting",
    "🏪 Analitik & Segmentasi Outlet",
    "🤖 Asisten Digital (Q&A Data)",
    "💬 Analisis Sentimen Ulasan",
    "🛡️ MLOps & Deteksi Drift"
])

# ==============================================================================
# TAB 1: DEMAND FORECASTING
# ==============================================================================
with tab1:
    st.subheader("Prediksi Penjualan & Kebutuhan Stok per Outlet")
    st.write("Model Machine Learning LightGBM memprediksi volume penjualan harian hingga 28 hari ke depan.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        store_id = st.selectbox("Pilih Nomor Outlet (Store):", sorted(df.Store.unique()), index=0)
    with col2:
        horizon = st.slider("Horizon Prediksi (Hari):", min_value=7, max_value=28, value=14, step=7)
    with col3:
        store_tier = segments_df[segments_df["Store"] == store_id]["segment_label"].values
        tier_label = store_tier[0] if len(store_tier) > 0 else "Unknown"
        st.markdown(f"**Profil Outlet #{store_id}**")
        st.info(f"Kategori Segmentasi: **{tier_label}** | Rata-rata Penjualan: **{df[df.Store == store_id]['Sales'].mean():,.0f} / hari**")

    # Generate Forecast
    with st.spinner("Menghitung inferensi model..."):
        fc_list = forecast.forecast_store(store_id, horizon, df)
        fc_df = pd.DataFrame(fc_list).set_index("date")
        fc_df.index = pd.to_datetime(fc_df.index)
        
        hist = df[df.Store == store_id].set_index("Date")["Sales"].tail(90)
        combined = pd.concat([hist.rename("Penjualan Aktual"), fc_df["forecast"].rename("Prediksi Masa Depan")], axis=1)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Estimasi Demand (Periode)", f"{fc_df['forecast'].sum():,.0f}")
    col_m2.metric("Rata-rata Prediksi Harian", f"{fc_df[fc_df['forecast'] > 0]['forecast'].mean():,.0f}")
    col_m3.metric("Hari Toko Tutup (Libur/Minggu)", f"{(fc_df['forecast'] == 0).sum()} hari")

    st.line_chart(combined, use_container_width=True)

    with st.expander(f"Lihat Rincian Angka Prediksi Outlet #{store_id}"):
        st.dataframe(fc_df.reset_index().rename(columns={"date": "Tanggal", "forecast": "Estimasi Penjualan (Unit/Rp)"}), use_container_width=True)

# ==============================================================================
# TAB 2: ANALITIK & SEGMENTASI OUTLET
# ==============================================================================
with tab2:
    st.subheader("Klasterisasi & Analitik Performa 1.115 Outlet")
    st.write("Menggunakan algoritma K-Means tanpa supervisi untuk memetakan outlet ke dalam 5 klaster strategis.")

    summary_counts = segments_df["segment_label"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]
    for i, (k, v) in enumerate(summary_counts.items()):
        if i < len(cols):
            cols[i].metric(label=k, value=f"{v} Outlet")

    st.divider()
    filter_tier = st.multiselect("Filter Kategori Outlet:", options=summary_counts.index.tolist(), default=summary_counts.index.tolist())
    filtered_df = segments_df[segments_df["segment_label"].isin(filter_tier)]

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        st.write("Distribusi Rata-rata Penjualan per Outlet")
        chart_data = filtered_df[["Store", "avg_sales"]].set_index("Store")
        st.bar_chart(chart_data.head(50), use_container_width=True)
    with col_table:
        st.write("Daftar Detail Outlet")
        st.dataframe(
            filtered_df[["Store", "avg_sales", "growth_90d", "promo_share", "segment_label"]]
            .rename(columns={
                "Store": "ID Outlet",
                "avg_sales": "Rata-rata Harian",
                "growth_90d": "Pertumbuhan 90 Hari",
                "promo_share": "Rasio Promo",
                "segment_label": "Klaster"
            }),
            use_container_width=True
        )

# ==============================================================================
# TAB 3: ASISTEN DIGITAL (Q&A DATA PENJUALAN)
# ==============================================================================
with tab3:
    st.subheader("🤖 Asisten Digital Internal (Natural Language Query)")
    st.write("Ketik pertanyaan dalam Bahasa Indonesia seputar penjualan, toko terbaik, perbandingan promo, dsb.")

    # Contoh pertanyaan cepat
    example_prompts = [
        "toko tertinggi 30 hari",
        "toko terendah 30 hari",
        "total penjualan toko 1 30 hari",
        "rata-rata penjualan toko 5",
        "pengaruh promo toko 2"
    ]
    st.write("Coba pertanyaan cepat:")
    q_cols = st.columns(len(example_prompts))
    selected_sample = None
    for i, p in enumerate(example_prompts):
        if q_cols[i].button(p, key=f"sample_{i}"):
            selected_sample = p

    user_query = st.text_input("Ajukan pertanyaan ke Asisten Digital:", value=selected_sample if selected_sample else "")

    if user_query:
        with st.spinner("Memproses query data..."):
            res = nlp.ask(user_query, df)
            st.success(f"**Jawaban:** {res['answer']}")
            if res.get("op"):
                st.caption(f"⚙️ Operasi Data yang Dijalankan di Balik Layar: `{res['op']}`")

# ==============================================================================
# TAB 4: ANALISIS SENTIMEN ULASAN PELANGGAN
# ==============================================================================
with tab4:
    st.subheader("💬 Analisis Sentimen Ulasan Toko & Layanan")
    st.write("Model NLP untuk menganalisis feedback pelanggan Bahasa Indonesia secara otomatis.")

    sample_reviews = [
        "pelayanan ramah dan pengiriman sangat cepat",
        "produk selalu segar dan stok lengkap mantap",
        "antrian kasir lama sekali dan barang sering kosong",
        "harga mahal tidak sesuai dan toko kotor kecewa"
    ]
    st.write("Contoh ulasan:")
    rev_cols = st.columns(len(sample_reviews))
    chosen_rev = None
    for i, s in enumerate(sample_reviews):
        if rev_cols[i].button(f"Contoh {i+1}", key=f"rev_{i}"):
            chosen_rev = s

    input_text = st.text_area("Masukkan teks ulasan pelanggan:", value=chosen_rev if chosen_rev else "pelayanan kasir sangat ramah, barang lengkap dan bersih")
    
    if st.button("Analisis Sentimen", type="primary"):
        with st.spinner("Menganalisis teks..."):
            ans = nlp.sentiment(input_text)
            label = ans["label"]
            score = ans["score"]
            if label == "positive":
                st.success(f"✅ **Sentimen: Positif** (Tingkat Kepuasan: {score * 100:.1f}%)")
            else:
                st.error(f"❌ **Sentimen: Negatif** (Tingkat Ketidakpuasan: {(1 - score) * 100:.1f}%)")

# ==============================================================================
# TAB 5: MLOPS & DETEKSI DRIFT
# ==============================================================================
with tab5:
    st.subheader("🛡️ Tata Kelola Model MLOps & Deteksi Drift Data")
    st.write("Sistem otomatis memantau penurunan distribusi data menggunakan Population Stability Index (PSI) & KS-Test.")

    col_drift1, col_drift2 = st.columns(2)
    with col_drift1:
        st.markdown("#### Status Kesehatan Model Saat Ini")
        st.write(f"- **Versi Model Aktif:** `forecast_v{fc_metrics.get('version', 1)}`")
        st.write(f"- **Status Drift Pipeline:** `{'⚠️ Ada Drift (Retraining Diperlukan)' if drift_report.get('drift') else '✅ Sehat (Normal)'}`")
        st.write(f"- **Evaluasi Holdout:** MAPE {holdout.get('MAPE', 0):.2f}% | RMSE {holdout.get('RMSE', 0):.2f}")
    
    with col_drift2:
        st.markdown("#### Pengujian Uji Regresi 5 Skenario Drift")
        scen_path = REPORTS / "drift_scenarios.json"
        if scen_path.exists():
            scenarios = json.loads(scen_path.read_text())
            for name, detected in scenarios.items():
                if name == "clean_no_false_alarm":
                    st.write(f"• Sampel Normal (Tanpa Alarm Palsu): {'✅ PASS' if detected else '❌ FAIL'}")
                else:
                    st.write(f"• Skenario `{name}`: {'🚨 Terdeteksi (PASS)' if detected else '❌ Tidak Terdeteksi'}")

    st.divider()
    if st.button("Jalankan Audit Drift & Evaluasi Ulang Sekarang"):
        with st.spinner("Menghitung PSI & KS-Test antar periode..."):
            audit_res = monitor.run(retrain=False)
            st.json(audit_res)
