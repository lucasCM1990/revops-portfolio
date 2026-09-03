# Piece 1 — Customer Health & Churn Risk

Retention / CS Ops side of the portfolio. A curated public dataset (Kaggle mirror
of the BCG X Data Science Job Simulation case — a B2B/SME energy retailer,
referred to in the data as "PowerCo") standing in for an operational churn
analysis. Not real employer data.

## What's here

- `notebooks/01_churn_model.py` — cleans the raw data, engineers client + price
  features, trains a baseline Random Forest churn model, ranks active accounts
  by relative risk, and computes GRR / churn rate.
- `notebooks/02_driver_analysis.py` — answers five specific retention questions
  (sales channel, tenure, product count, dual-fuel bundling, acquisition
  campaign quality) directly from the raw data.
- `dashboard/app.py` — a Dash app ("Retention Book") presenting the results:
  KPI cards, driver charts, and a searchable/filterable account book.
- `data/clean/` — the CSV outputs of both scripts (small; committed).
- `data/raw/` — not committed; see `SOURCE.md` for exactly where to download it.

## Running it

```bash
# 1. Download the raw data — see data/raw/SOURCE.md
# 2. From this folder:
pip install pandas numpy scikit-learn plotly dash dash-bootstrap-components dash-ag-grid
python3 notebooks/01_churn_model.py
python3 notebooks/02_driver_analysis.py
cd dashboard && python3 app.py
# open http://127.0.0.1:8050
```

## Method, honestly

The baseline model (Random Forest, AUC 0.63) has weak-to-moderate discrimination —
adding a full year of price history barely moved AUC (0.595 → 0.599), which
mirrors the original BCG case's own finding that price sensitivity alone doesn't
explain most churn here. Because of that, the model is used only to **rank**
accounts by relative risk (health score = a percentile, not a probability) — it
never produces an absolute dollar forecast. Every dollar figure in the dashboard
(GRR, margin at risk, margin by driver) is real, unmodeled margin; only the
ordering comes from the model.
