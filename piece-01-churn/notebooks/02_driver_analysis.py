"""
Piece 1 - Driver analysis
Answers 5 business questions the raw client_data.csv can actually support:

1. Does the sales channel affect churn?
2. Does tenure protect against churn?
3. Do multi-product accounts churn less?
4. Are dual-fuel (electricity + gas) accounts stickier?
5. Which acquisition campaign brings in the most valuable (highest-margin) accounts?

channel_sales and origin_up are anonymized hash codes in the source data (no real
channel/campaign names exist) -- relabeled here as "Channel A/B/C..." and
"Campaign A/B/C..." sorted by account volume, purely for readability. This is a
labeling choice, not new data.
"""
import pandas as pd

RAW = 'data/raw/client_data.csv'
CLEAN_DIR = 'data/clean'

df = pd.read_csv(RAW)

# ---------- relabel anonymized codes by volume, for a readable axis ----------
def relabel(series, prefix):
    order = series.value_counts().index.tolist()
    mapping = {code: f'{prefix} {chr(65 + i)}' if i < 26 else f'{prefix} {i+1}' for i, code in enumerate(order)}
    return series.map(mapping)

df['channel_label'] = relabel(df['channel_sales'], 'Channel')
df['campaign_label'] = relabel(df['origin_up'], 'Campaign')

# ---------- 1. churn by sales channel ----------
by_channel = df.groupby('channel_label').agg(
    n_accounts=('id', 'count'), churn_rate=('churn', 'mean')
).reset_index().sort_values('n_accounts', ascending=False)
by_channel['dimension'] = 'channel'
by_channel = by_channel.rename(columns={'channel_label': 'group'})

# ---------- 2. churn by tenure ----------
by_tenure = df.groupby('num_years_antig').agg(
    n_accounts=('id', 'count'), churn_rate=('churn', 'mean')
).reset_index().sort_values('num_years_antig')
by_tenure['dimension'] = 'tenure'
by_tenure = by_tenure.rename(columns={'num_years_antig': 'group'})
by_tenure['group'] = by_tenure['group'].astype(str)

# ---------- 3. churn by product count ----------
df['product_bucket'] = df['nb_prod_act'].clip(upper=4).map({1: '1', 2: '2', 3: '3', 4: '4+'})
by_products = df.groupby('product_bucket').agg(
    n_accounts=('id', 'count'), churn_rate=('churn', 'mean')
).reset_index().sort_values('product_bucket')
by_products['dimension'] = 'products'
by_products = by_products.rename(columns={'product_bucket': 'group'})

# ---------- 4. churn by dual-fuel status ----------
df['fuel_label'] = df['has_gas'].map({'t': 'Electricity + gas', 'f': 'Electricity only'})
by_fuel = df.groupby('fuel_label').agg(
    n_accounts=('id', 'count'), churn_rate=('churn', 'mean')
).reset_index()
by_fuel['dimension'] = 'fuel'
by_fuel = by_fuel.rename(columns={'fuel_label': 'group'})

churn_drivers = pd.concat([by_channel, by_tenure, by_products, by_fuel], ignore_index=True)
churn_drivers = churn_drivers[['dimension', 'group', 'n_accounts', 'churn_rate']]
churn_drivers.to_csv(f'{CLEAN_DIR}/churn_drivers.csv', index=False)

# ---------- 5. campaign value: which acquisition channel brings the best accounts ----------
campaign_value = df.groupby('campaign_label').agg(
    n_accounts=('id', 'count'), avg_margin=('net_margin', 'mean'), churn_rate=('churn', 'mean')
).reset_index().sort_values('n_accounts', ascending=False)
campaign_value = campaign_value.rename(columns={'campaign_label': 'campaign'})
campaign_value.to_csv(f'{CLEAN_DIR}/campaign_value.csv', index=False)

# ---------- print findings so they can be checked before wiring into the dashboard ----------
print("=== Churn by sales channel ===")
print(by_channel[['group', 'n_accounts', 'churn_rate']].to_string(index=False))
print("\n=== Churn by tenure (years) ===")
print(by_tenure[['group', 'n_accounts', 'churn_rate']].to_string(index=False))
print("\n=== Churn by product count ===")
print(by_products[['group', 'n_accounts', 'churn_rate']].to_string(index=False))
print("\n=== Churn by fuel type ===")
print(by_fuel[['group', 'n_accounts', 'churn_rate']].to_string(index=False))
print("\n=== Campaign value (avg margin, sorted by volume) ===")
print(campaign_value.to_string(index=False))

print(f"\nExported: {CLEAN_DIR}/churn_drivers.csv ({len(churn_drivers)} rows)")
print(f"Exported: {CLEAN_DIR}/campaign_value.csv ({len(campaign_value)} rows)")
