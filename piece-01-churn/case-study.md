# Customer Health & Churn Risk — PowerCo (B2B Energy)

**Live dashboard:** [revops-portfolio.onrender.com](https://revops-portfolio.onrender.com) · **Code:** [github.com/lucasCM1990/revops-portfolio](https://github.com/lucasCM1990/revops-portfolio)

A retention / CS Ops analysis of a B2B energy retailer's customer book: who is at risk, why, and what to do about each finding — not just a churn dashboard.

**The concrete insight:** accounts that already churned were higher-margin on average ($228) than the accounts that stayed ($185) — the customers leaving are not the marginal ones. That single number is why this piece ranks accounts by risk *and* value instead of just counting logos. A second, GTM-facing one: the acquisition channel bringing in the most customers (Channel/Campaign A) also has the worst retention (12.6% vs. 6.0% for Channel B) — volume and quality of acquisition are not the same thing here.

---

## What

An end-to-end churn risk analysis for PowerCo, a mid-market gas and electricity retailer serving business (SME/enterprise) customers: data cleaning and feature engineering, a baseline risk model used strictly to *rank* accounts (not to forecast a dollar figure), a driver analysis answering five concrete retention questions, and a per-account recommendation layer — each finding paired with a 5W2H action (What/Why/Who/When/Where/How/How much) and, where the action is an ongoing process rather than a one-time fix, a PDCA follow-up flag.

## Why

Churn rate alone understates what's at stake here: **accounts that already churned were higher-margin on average ($228) than the accounts that stayed ($185)**. A retention program that only counts logos, not margin, would miss that the customers actually walking out the door are disproportionately the valuable ones. This piece exists to answer the question a CS Ops or RevOps leader actually has: not "how many left," but "how much revenue is at risk, where is it concentrated, and what do we do about it."

## Who

Built for a **Revenue / Sales Operations Analyst** audience — recruiters and hiring managers at B2B companies with a retention motion (SaaS, subscription, or repeat-purchase B2B). The recommended actions name the departments that would actually own them: Customer Success, Account Management, Sales Ops, Marketing, and Finance/RevOps — because a churn analysis that doesn't say who acts on it isn't finished.

## Where / When

- **Population:** 14,606 B2B/SME energy accounts (13,187 still active as of the snapshot).
- **Data window:** a single point-in-time snapshot with a forward-looking churn flag (did the account cancel within the next 3 months), plus a full year of monthly price history (Jan–Dec 2015) per account.
- **Source:** a curated public sample — a Kaggle mirror of the BCG X Data Science Job Simulation case. **Not operational data from any employer.** See `data/raw/SOURCE.md` for exact dataset links and licenses.

## How

1. **Clean & engineer** (`notebooks/01_churn_model.py`): dropped a redundant column (99.99% identical to another), treated an anonymized "MISSING" sales-channel code as its own category rather than a null, and engineered tenure/contract-timing/consumption features plus aggregated price-history features (mean, volatility, year-over-year change).
2. **Model, honestly**: tried logistic regression first (AUC 0.595), then added a full year of price history (AUC 0.599 — barely moved, meaning price sensitivity alone doesn't explain churn here, which mirrors the original BCG case's own conclusion). Random Forest did better (AUC 0.634) but is still weak-to-moderate — so the model is used **only to rank** accounts by relative risk, never to produce an absolute probability or dollar forecast. Health score is a **percentile** of that ranking (100 = safest in this book), not `1 − churn probability` — the model's `class_weight='balanced'` setting (needed for ranking quality on an imbalanced target) inflates raw probabilities well past the true ~9.7% base rate, and using them directly would have mislabeled most of the book as "at risk."
3. **Driver analysis** (`notebooks/02_driver_analysis.py`): answered five questions directly from the raw data — sales channel, tenure, product count, dual-fuel bundling, and acquisition-campaign quality — independent of the model, so these findings don't inherit its limitations.
4. **Per-account explainability** (Account Detail tab): for any single account, the three features where it deviates most from the book average (a z-score comparison against the model's top features) — not SHAP, deliberately: a shallow baseline model doesn't warrant that level of claimed rigor, and a simpler, honestly-labeled comparison is easier to defend than a more sophisticated technique bolted onto a model this weak.
5. **Present** (`dashboard/app.py`, Dash + Plotly): four tabs — Overview (KPIs, risk concentration, top findings), Drivers (five questions, each with a 5W2H recommended action), Account Book (searchable/filterable grid of all 13,187 active accounts), Account Detail (per-account risk explanation and recommendation).

## How much

| Metric | Value | Basis |
|---|---|---|
| Churn rate | 9.7% | Observed, full historical book |
| GRR (margin-based) | 88.3% | Real margin retained vs. starting book — no model involved |
| Margin lost to churn | $324,046 | Sum of net margin on churned accounts |
| Top-decile risk concentration | 17.9% of active margin in the top 10% riskiest accounts | Ranking-based, not a probability-weighted estimate |
| Channel A excess-churn cost | ~$88,100/year | 443 "excess" churned accounts vs. Channel C's rate × Channel A's average margin |
| Years 3–4 tenure cohort, margin already lost | $217,901 | Historical churn within that cohort |
| Dual-fuel cross-sell upside | ~$87,000/year in margin alone | Average margin gap ($249 vs. $176) × the 11,955-account electricity-only pool, before counting the churn-rate improvement |

Where a number isn't available (the Campaign A/B budget-reallocation "how much" would need a customer-lifetime model this snapshot dataset can't support), the case says so explicitly rather than guessing.

## How (decision)

Each driver finding ends in a named action, not a chart:

- **Channel A churns at 12.1%** (vs. 9.7% average, vs. 5.6% for Channel C, which also has *higher* margin) → Sales Ops + CS audit Channel A's qualification and first-90-day onboarding against Channel C's.
- **Churn peaks in years 3–4, drops at year 6** → CS triggers an automated health-check/renewal conversation at month 30, before the risk window, not after.
- **Product count doesn't reduce churn** (10.0% at 1 product vs. 8.7% at 4+) → RevOps corrects internal messaging: cross-sell is a revenue lever here, not a retention one — stop selling it internally as both.
- **Dual-fuel accounts churn less *and* carry more margin** → Marketing + Sales launch a gas cross-sell campaign targeted at electricity-only accounts, prioritized by risk score.
- **Campaign A brings volume, Campaign B brings retention** → Marketing + Finance build an LTV-adjusted comparison before the next acquisition budget cycle, instead of allocating by volume alone.

The Account Detail tab applies the same logic one account at a time: real deviation from the book average → a specific, rule-based recommended action → a real dollar figure (that account's own margin), so a CS rep opening a specific record gets a starting point, not just a score.

---

*Provenance: curated public dataset (Kaggle mirror of the BCG X Data Science Job Simulation case). No data from any current or former employer appears anywhere in this piece.*
