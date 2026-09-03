# Raw data sources

Raw CSVs are not committed to this repo (kept out via `.gitignore`) to keep it light.
Download them from Kaggle and place them in this folder to reproduce the pipeline.

## client_data.csv

- Source: [PowerCo Churn Data](https://www.kaggle.com/datasets/dharun4772/powerco-churn-data) (Kaggle, mirror of the BCG X Data Science Job Simulation dataset)
- License: MIT
- 14,606 rows, 26 columns — one row per client company

## price_data.csv

- Source: [PowerCO. (a major gas and electricity utility)](https://www.kaggle.com/datasets/takusingh/powerco-a-major-gas-and-electricity-utility) (Kaggle)
- License: CC BY-SA 4.0
- 193,002 rows, 8 columns — monthly price history per client, Jan–Dec 2015

## Provenance note

This is a curated public sample (a well-known BCG X case-study dataset mirrored on
Kaggle), not operational data from any employer. See `../notebooks/01_churn_model.py`
for how it's cleaned and modeled.
