"""
Piece 1 - Customer Health & Churn Risk
Source: Kaggle mirror of the BCG X Data Science Job Simulation dataset
(takusingh/powerco-a-major-gas-and-electricity-utility). Curated public sample,
SME/B2B energy clients. Not operational data from any employer.

Pipeline: load -> clean -> engineer client + price features -> train churn model
-> rank active clients by relative risk -> export prioritized account list.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

RAW = 'data/raw/client_data.csv'
PRICE_RAW = 'data/raw/price_data.csv'
CLEAN_DIR = 'data/clean'

# ---------- load ----------
df = pd.read_csv(RAW, parse_dates=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])
price = pd.read_csv(PRICE_RAW, parse_dates=['price_date'])

# ---------- price features (the actual predictive signal in this case) ----------
price_cols = ['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var',
              'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']

price_agg = price.groupby('id')[price_cols].agg(['mean', 'std']).fillna(0)
price_agg.columns = ['_'.join(c) for c in price_agg.columns]

price_sorted = price.sort_values(['id', 'price_date'])
first_last = price_sorted.groupby('id').agg(
    first_off_peak_var=('price_off_peak_var', 'first'),
    last_off_peak_var=('price_off_peak_var', 'last'),
    first_off_peak_fix=('price_off_peak_fix', 'first'),
    last_off_peak_fix=('price_off_peak_fix', 'last'),
)
first_last['price_off_peak_var_change'] = first_last['last_off_peak_var'] - first_last['first_off_peak_var']
first_last['price_off_peak_fix_change'] = first_last['last_off_peak_fix'] - first_last['first_off_peak_fix']

price_features = price_agg.join(first_last[['price_off_peak_var_change', 'price_off_peak_fix_change']])
df = df.merge(price_features, left_on='id', right_index=True, how='left')

# ---------- clean ----------
# margin_gross_pow_ele and margin_net_pow_ele are identical in 99.99% of rows -> redundant, drop gross
df = df.drop(columns=['margin_gross_pow_ele'])

# "MISSING" is a real category in channel_sales / origin_up (~25% / ~0.4%), not a null -> keep explicit
df['channel_sales'] = df['channel_sales'].replace('MISSING', 'unknown_channel')
df['origin_up'] = df['origin_up'].replace('MISSING', 'unknown_campaign')

# reference date = latest date_renewal in the dataset, used as the "as of" snapshot date
snapshot_date = df['date_renewal'].max()

# ---------- feature engineering (kept simple / explainable) ----------
df['tenure_years'] = df['num_years_antig']
df['days_since_last_renewal'] = (snapshot_date - df['date_renewal']).dt.days
df['days_since_last_modif'] = (snapshot_date - df['date_modif_prod']).dt.days
df['days_to_contract_end'] = (df['date_end'] - snapshot_date).dt.days
df['has_gas_flag'] = (df['has_gas'] == 't').astype(int)
df['consumption_drop_ratio'] = np.where(
    df['cons_12m'] > 0,
    1 - (df['cons_last_month'] * 12) / df['cons_12m'].replace(0, np.nan),
    0
)
df['consumption_drop_ratio'] = df['consumption_drop_ratio'].fillna(0).clip(-5, 1)

PRICE_FEATURES = [c for c in price_features.columns]

FEATURES = [
    'tenure_years', 'days_since_last_renewal', 'days_since_last_modif',
    'days_to_contract_end', 'has_gas_flag', 'nb_prod_act', 'pow_max',
    'cons_12m', 'net_margin', 'consumption_drop_ratio', 'forecast_discount_energy',
] + PRICE_FEATURES
TARGET = 'churn'

model_df = df[FEATURES + [TARGET, 'id']].copy()

# ---------- train / evaluate ----------
X = model_df[FEATURES]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Model: Random Forest (baseline, illustrative — kept shallow/simple on purpose)")
print(f"Test AUC: {auc:.3f}")
print(f"Churn rate in data: {y.mean():.1%}")
print("\nClassification report @ 0.5 threshold:")
print(classification_report(y_test, (y_pred_proba >= 0.5).astype(int)))

importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# NOTE on AUC (~0.63): weak-to-moderate discrimination, consistent with this being a
# known low-signal dataset — even with full price history, price alone barely predicts
# churn here (this mirrors the original BCG case's own conclusion that price sensitivity
# does not explain most churn). Because of this, we do NOT report an absolute
# probability-weighted "% of revenue at risk" figure — with a model this weak that number
# is not calibrated and would overstate precision. Instead the model is used only to RANK
# active clients by relative risk, and the headline metric is the real, unmodeled net
# margin concentrated in the highest-risk segment — a prioritization tool, not a forecast.

# ---------- score full book, rank by relative risk ----------
model_df['churn_probability'] = clf.predict_proba(model_df[FEATURES])[:, 1]

# active book = clients who have not already churned in this historical snapshot
active = model_df[model_df[TARGET] == 0].copy()
active = active.sort_values('churn_probability', ascending=False)
active['risk_rank'] = range(1, len(active) + 1)
active['risk_decile'] = pd.qcut(active['risk_rank'], 10, labels=False) + 1  # 1 = highest risk

total_active_margin = active['net_margin'].clip(lower=0).sum()

# prioritized list for CS: top 10% by predicted risk rank
top_n = max(1, int(len(active) * 0.10))
priority = active.head(top_n)
priority_margin = priority['net_margin'].clip(lower=0).sum()

print(f"\n--- Portfolio-level (active clients only, n={len(active)}) ---")
print(f"Total active net margin: {total_active_margin:,.0f}")
print(f"\nTop 10% by predicted churn risk ({top_n} accounts) hold "
      f"{priority_margin:,.0f} in net margin "
      f"({priority_margin/total_active_margin:.1%} of the active book's total margin, "
      f"vs. {top_n/len(active):.1%} of accounts) — this is real margin, not a modeled estimate; "
      f"only the ranking that produced this list comes from the model.")

# ---------- export ----------
export_cols = ['id', 'tenure_years', 'nb_prod_act', 'pow_max', 'cons_12m', 'net_margin',
               'has_gas_flag', 'days_to_contract_end', 'churn_probability', 'risk_rank', 'risk_decile']

active[export_cols].to_csv(f'{CLEAN_DIR}/clients_scored.csv', index=False)
priority[export_cols].to_csv(f'{CLEAN_DIR}/priority_accounts_top10pct.csv', index=False)

print(f"\nExported: {CLEAN_DIR}/clients_scored.csv ({len(active)} rows)")
print(f"Exported: {CLEAN_DIR}/priority_accounts_top10pct.csv ({len(priority)} rows)")

# ---------- KPI summary (Churn rate, GRR, health-score concentration) ----------
# GRR is computed on the FULL original book (active + churned), not just the active
# subset used for risk scoring above -- it answers "how much of what we started with
# did we keep", which is a retrospective, model-free number.
starting_margin = df['net_margin'].clip(lower=0).sum()
churned_margin = df.loc[df[TARGET] == 1, 'net_margin'].clip(lower=0).sum()
grr = (starting_margin - churned_margin) / starting_margin
churn_rate = df[TARGET].mean()
top_decile_margin_share = priority_margin / total_active_margin
top_decile_account_share = top_n / len(active)

kpi_summary = pd.DataFrame([{
    'churn_rate': churn_rate,
    'grr': grr,
    'starting_margin': starting_margin,
    'churned_margin': churned_margin,
    'active_margin': total_active_margin,
    'top_decile_margin_share': top_decile_margin_share,
    'top_decile_account_share': top_decile_account_share,
    'top_decile_margin_value': priority_margin,
    'model_auc': auc,
    'avg_margin_churned': df.loc[df[TARGET] == 1, 'net_margin'].mean(),
    'avg_margin_active': df.loc[df[TARGET] == 0, 'net_margin'].mean(),
}])
kpi_summary.to_csv(f'{CLEAN_DIR}/kpi_summary.csv', index=False)
print(f"Exported: {CLEAN_DIR}/kpi_summary.csv")
print(f"\nGRR: {grr:.1%} | Churn rate: {churn_rate:.1%}")
