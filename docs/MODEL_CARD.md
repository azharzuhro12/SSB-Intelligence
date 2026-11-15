# Model Card: Store Sales Forecaster (Production Grade)
- **Model Type:** LightGBM Regressor (400 estimators, num_leaves=63, learning_rate=0.05).
- **Domain:** Retail & Distribution daily store sales forecasting (28-day ahead planning).
- **Training Data:** Rossmann Store Sales official dataset (1,017,209 records, 1,115 stores, 2013-01 to 2015-07).
- **Engineered Features:** Day of week, month, day, ISO week, StoreType, Assortment, CompetitionDistance, Promo, StateHoliday, non-leaking historical sales lags (28, 35, 56 days), and rolling means (7-day, 28-day).
- **Validation Scheme:** Rolling-origin time-series cross-validation (3 historical folds) + strict out-of-time holdout (last 28 calendar days).
- **Evaluation Results:**
  - Holdout MAPE: **9.36%** (vs Baseline Seasonal-Naive: 11.96%)
  - Holdout WAPE: **8.94%** (vs Baseline: 11.86%)
  - Holdout RMSE: **859.51** (vs Baseline: 1218.02)
- **Governance & Monitoring:** Automatic drift alerts via PSI (threshold > 0.25). Model promotion only occurs if holdout MAPE improves or matches previous registry champion.

# Model Card: Indonesian Review Sentiment Classifier
- **Model Type:** Scikit-Learn TF-IDF (unigrams + bigrams) + Logistic Regression.
- **Inference Latency:** < 5 ms on CPU.
- **Performance:** Macro-F1: 0.92 on stratified test split.
