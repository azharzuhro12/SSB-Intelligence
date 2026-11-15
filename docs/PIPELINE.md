# Pipeline Documentation
```mermaid
flowchart LR
 A[data.load] --> B[etl.validate] --> C[clean + features] --> D[(data/processed/features.pkl)]
 D --> E[forecast.train] --> F[registry.json + joblib]
 D --> G[outlet.segment] --> H[reports/outlet_segments.csv]
 D --> I[monitor.run] -->|drift| E
 F --> J[api.py] --> K[dashboard.py]
```
- **Validation (fails run):** non-null keys, no duplicate (Store, Date), Sales ≥ 0, Promo ∈ {0,1}.
- **Idempotency:** outputs overwritten; synthetic data seeded.
- **Scheduling:** `python -c "from ssb import monitor; monitor.run()"` via cron (e.g. weekly).
- **Registry:** each training adds a version; `current` changes only if holdout MAPE does not worsen.
