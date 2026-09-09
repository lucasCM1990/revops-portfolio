"""
Piece 1 - Customer Health & Churn Risk (PowerCo, SME/B2B energy)

Visual design ported from Lucas's Claude Design mockup (Desktop/Retention Book.html)
-- the "blueprint" system: Barlow/Barlow Condensed, navy/blue-gray tokens, sharp
corners, plus-sign corner marks. Numbers here are real, computed from
notebooks/01_churn_model.py -- the mockup used illustrative placeholder figures
(e.g. "12 accounts, $426k MRR") that do not appear anywhere in this file.

Three tabs: Overview (KPIs, charts, risk queue, findings), Drivers (five business
questions answered from the raw data -- channel, tenure, product count,
dual-fuel bundling, campaign quality), and Account Book (searchable/filterable
grid of all active accounts). A fourth tab from the mockup, Account Detail
(per-account drill-down), is not built yet.

Data: curated public sample (Kaggle mirror of the BCG X Data Science Job
Simulation dataset), not operational data from any employer. Risk score is an
illustrative baseline model (Random Forest, AUC 0.63) used only to rank accounts
by relative risk -- see notebooks/01_churn_model.py for the full method and its
limitations.
"""
import os
import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import Dash, dcc, html, Input, Output, State

# resolved relative to this file, not the working directory the process is
# started from -- matters once this runs under gunicorn on a host like Render
CLEAN_DIR = Path(__file__).resolve().parent.parent / 'data' / 'clean'

clients = pd.read_csv(CLEAN_DIR / 'clients_scored.csv')
kpi = pd.read_csv(CLEAN_DIR / 'kpi_summary.csv').iloc[0]
churn_drivers = pd.read_csv(CLEAN_DIR / 'churn_drivers.csv')
campaign_value = pd.read_csv(CLEAN_DIR / 'campaign_value.csv')

# small groups are noise, not signal -- exclude from charts (kept in the CSV export)
MIN_N = 50
churn_drivers_chart = churn_drivers[churn_drivers['n_accounts'] >= MIN_N]
campaign_value_chart = campaign_value[campaign_value['n_accounts'] >= MIN_N]
overall_churn = kpi['churn_rate']

# ---------- health score + bands (derived from the risk model, all real) ----------
# NOTE: health_score is a PERCENTILE of predicted risk (100 = safest, 0 = riskiest
# in this book), not (1 - churn_probability). The model uses class_weight='balanced'
# for ranking quality, which inflates raw probabilities well above the true ~9.7%
# base rate -- using them directly would mislabel most of the book as "at risk".
# Percentile rank has no such calibration problem: it is uniform by construction and
# only claims relative standing, consistent with how the model is used everywhere
# else in this piece (ranking, not forecasting).
clients = clients.sort_values('churn_probability', ascending=True).reset_index(drop=True)
clients['health_score'] = (100 * (1 - clients.index / (len(clients) - 1))).round().astype(int)

def band_for(score):
    if score < 40:
        return 'Critical'
    if score < 65:
        return 'At risk'
    if score < 85:
        return 'Stable'
    return 'Strong'

clients['health_band'] = clients['health_score'].apply(band_for)
BAND_ORDER = ['Critical', 'At risk', 'Stable', 'Strong']
BAND_TAG_CLASS = {
    'Critical': 'rb-tag rb-tag-critical',
    'At risk': 'rb-tag rb-tag-atrisk',
    'Stable': 'rb-tag rb-tag-stable',
    'Strong': 'rb-tag rb-tag-strong',
}

band_summary = clients.groupby('health_band').agg(
    accounts=('id', 'count'), margin=('net_margin', 'sum')
).reindex(BAND_ORDER).reset_index()
total_margin = clients['net_margin'].sum()
band_summary['margin_share'] = band_summary['margin'] / total_margin

avg_health_score = (clients['health_score'] * clients['net_margin']).sum() / clients['net_margin'].sum()
below_65 = clients[clients['health_score'] < 65]
margin_at_risk = below_65['net_margin'].sum()
margin_at_risk_share = margin_at_risk / total_margin

# ---------- colors + chart template (from the mockup's :root tokens) ----------
NAVY = '#1d2d3d'
ACCENT = '#5980a6'
ACCENT_LIGHT = '#94bce3'
NEUTRAL = '#98989b'
GRID = 'rgba(29,31,32,0.10)'

rb_template = go.layout.Template()
rb_template.layout = go.Layout(
    font=dict(family='Barlow, sans-serif', size=13, color='#1d1f20'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    margin=dict(l=50, r=20, t=20, b=40),
)
pio.templates['retention_book'] = rb_template
pio.templates.default = 'retention_book'

# ---------- chart 1: net margin by risk decile ----------
decile_margin = clients.groupby('risk_decile')['net_margin'].sum().reset_index().sort_values('risk_decile')
decile_margin['risk_decile'] = decile_margin['risk_decile'].astype(int)
decile_margin['is_top_risk'] = decile_margin['risk_decile'] == 1

fig_decile = px.bar(
    decile_margin, x='risk_decile', y='net_margin', color='is_top_risk',
    color_discrete_map={True: NAVY, False: ACCENT_LIGHT},
    labels={'risk_decile': 'Risk decile (1 = highest predicted risk)', 'net_margin': 'Net margin ($)'},
)
fig_decile.update_layout(showlegend=False, height=320)

# ---------- chart 2: health band distribution ----------
fig_bands = px.bar(
    band_summary, x='health_band', y='accounts',
    color='health_band',
    color_discrete_map={'Critical': '#b23b3b', 'At risk': ACCENT, 'Stable': ACCENT_LIGHT, 'Strong': NAVY},
    labels={'health_band': '', 'accounts': 'Accounts'},
    category_orders={'health_band': BAND_ORDER},
)
fig_bands.update_layout(showlegend=False, height=320)

# ---------- driver charts: churn rate by group, with the book average as a reference line ----------
def driver_bar(dimension, x_label):
    data = churn_drivers_chart[churn_drivers_chart['dimension'] == dimension]
    fig = px.bar(
        data, x='group', y='churn_rate',
        labels={'group': x_label, 'churn_rate': 'Churn rate'},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(marker_color=ACCENT)
    fig.update_xaxes(type='category')  # group labels look numeric ("1","2"...) -- force category, not a numeric axis
    fig.update_layout(yaxis_tickformat='.0%', height=260, showlegend=False)
    fig.add_hline(
        y=overall_churn, line_dash='dot', line_color=NAVY,
        annotation_text=f'book average ({overall_churn:.1%})', annotation_position='top left',
        annotation_font_size=11,
    )
    return fig

fig_channel = driver_bar('channel', 'Sales channel')
fig_tenure = driver_bar('tenure', 'Years as a customer')
fig_products = driver_bar('products', 'Products contracted')
fig_fuel = driver_bar('fuel', 'Service type')

fig_campaign = px.bar(
    campaign_value_chart, x='campaign', y='avg_margin',
    labels={'campaign': 'Acquisition campaign', 'avg_margin': 'Average net margin ($)'},
)
fig_campaign.update_traces(marker_color=ACCENT)
fig_campaign.update_layout(height=260, showlegend=False)

def action_panel(action):
    """5W2H recommended-action block: what/why/who/when/where/how/how much,
    plus which departments own it and whether it warrants a PDCA follow-up
    cycle. This is business judgment built on the real numbers above it, not
    itself a number pulled from the data -- labelled as a recommendation, not
    a finding."""
    rows = [
        ('What', action['what']), ('Why', action['why']), ('Who', action['who']),
        ('When', action['when']), ('Where', action['where']), ('How', action['how']),
        ('How much', action['how_much']),
    ]
    return html.Div([
        html.Div([html.Span(label, className='rb-5w2h-label'), html.Span(value)], className='rb-5w2h-row')
        for label, value in rows
    ] + [
        html.Div(
            'PDCA follow-up recommended — this is an ongoing process change, not a one-time fix.'
            if action['pdca'] else
            'No PDCA cycle needed here — a single action to log and track (e.g. in CRM), not a repeatable process to monitor.',
            className='rb-pdca-note',
        ),
    ], className='rb-action-panel')

def question_card(question, chart, answer, action=None, card_id=None):
    children = [
        *corners(),
        html.Div(question, className='rb-question'),
        dcc.Graph(figure=chart, config={'displayModeBar': False}),
        html.Div(answer, className='rb-kpi-note', style={'marginTop': '4px'}),
    ]
    if action is not None:
        children += [
            dbc.Button(
                'Recommended action (5W2H) ▾', id=f'{card_id}-toggle', n_clicks=0,
                className='rb-action-toggle', color='link',
            ),
            dbc.Collapse(action_panel(action), id=f'{card_id}-collapse', is_open=False),
        ]
    return html.Div(children, className='rb-card blueprint', style={'marginBottom': '8px'})

# ---------- KPI cards ----------
def corners():
    return [html.Div(className=f'corner {c}') for c in ('tl', 'tr', 'bl', 'br')]

def kpi_card(label, value, note, dark=False):
    cls = 'rb-card blueprint' + (' rb-kpi-dark' if dark else '')
    return html.Div([
        *corners(),
        html.Div(label, className='rb-kpi-label'),
        html.Div(value, className='rb-kpi-value'),
        html.Div(note, className='rb-kpi-note'),
    ], className=cls)

kpi_row = dbc.Row([
    dbc.Col(kpi_card('Logo churn rate', f"{kpi['churn_rate']:.1%}", 'Observed, full historical book (14,606 accounts).'), md=3),
    dbc.Col(kpi_card('GRR — retained margin', f"{kpi['grr']:.1%}", 'Arithmetic on what already happened — no model calibration involved.'), md=3),
    dbc.Col(kpi_card('Average health score', f"{avg_health_score:.0f}/100", 'Margin-weighted across all active accounts. Model output, not an actual.'), md=3),
    dbc.Col(kpi_card(
        'Margin at risk',
        f"${margin_at_risk:,.0f}",
        f"{len(below_65):,} accounts scoring below 65 — {margin_at_risk_share:.0%} of active margin.",
        dark=True,
    ), md=3),
], className='g-3 mb-5')

# ---------- risk queue (top 8, numeric columns only -- no fabricated narrative) ----------
risk_queue = clients.sort_values('churn_probability', ascending=False).head(8).copy()
risk_queue_rows = []
for _, r in risk_queue.iterrows():
    risk_queue_rows.append(html.Tr([
        html.Td(r['id'][:12] + '…', style={'fontFamily': 'monospace', 'fontSize': '12.5px'}),
        html.Td(f"${r['net_margin']:,.0f}"),
        html.Td(str(int(r['health_score']))),
        html.Td(html.Span(r['health_band'], className=BAND_TAG_CLASS[r['health_band']])),
    ]))

risk_queue_table = html.Table([
    html.Thead(html.Tr([html.Th('Account'), html.Th('Net margin'), html.Th('Health'), html.Th('Band')])),
    html.Tbody(risk_queue_rows),
], className='table')

# ---------- findings (all traceable to the actual analysis) ----------
findings = [
    "Clients who already churned were higher-margin on average ($228) than clients who stayed ($185) — "
    "the accounts leaving are not the marginal ones.",
    "Price sensitivity barely predicts churn here: adding a full year of price history moved model AUC from "
    "0.595 to just 0.599. Tenure, consumption pattern and contract timing carry more signal than price.",
    f"The top 10% of active accounts by predicted risk hold {kpi['top_decile_margin_share']:.1%} of active "
    "margin — a real concentration effect, even though the underlying model is only weak-to-moderate (AUC 0.63).",
]
findings_list = html.Div([
    html.Div([html.Span(f'0{i+1}', className='rb-finding-num'), html.Span(f)], className='rb-finding')
    for i, f in enumerate(findings)
])

# ---------- Account Book: grid ----------
grid_columns = [
    {'field': 'id', 'headerName': 'Account', 'flex': 2},
    {'field': 'health_band', 'headerName': 'Band', 'flex': 1},
    {'field': 'health_score', 'headerName': 'Health', 'flex': 1},
    {'field': 'tenure_years', 'headerName': 'Tenure (yrs)', 'flex': 1},
    {'field': 'nb_prod_act', 'headerName': 'Products', 'flex': 1},
    {'field': 'net_margin', 'headerName': 'Net margin ($)', 'flex': 1,
     'valueFormatter': {'function': "d3.format(',.2f')(params.value)"}},
    {'field': 'days_to_contract_end', 'headerName': 'Days to contract end', 'flex': 1},
    {'field': 'churn_probability', 'headerName': 'Churn probability', 'flex': 1,
     'valueFormatter': {'function': "d3.format('.1%')(params.value)"}},
]

accounts_grid = dag.AgGrid(
    id='accounts-grid',
    columnDefs=grid_columns,
    rowData=clients.sort_values('health_score').to_dict('records'),
    defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
    columnSize='responsiveSizeToFit',
    className='ag-theme-quartz',
    style={'height': '560px'},
    dashGridOptions={'pagination': True, 'paginationPageSize': 20},
)

band_filter = html.Div([
    dbc.RadioItems(
        id='band-filter',
        className='rb-seg',
        inputClassName='form-check-input',
        labelClassName='form-check-label',
        options=[{'label': 'All', 'value': 'All'}] + [{'label': b, 'value': b} for b in BAND_ORDER],
        value='All',
    ),
])

# ---------- app ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Retention Book'
server = app.server  # the Flask instance gunicorn serves in production

header = html.Div([
    html.Div([
        html.Div('RETENTION BOOK', className='rb-title'),
        html.Div('CHURN · GRR · HEALTH SCORE', className='rb-subtitle'),
    ]),
    html.Div([
        html.Div(f"REFRESHED {datetime.date.today():%b %d, %Y}"),
        html.Div('SOURCE: RANDOM FOREST BASELINE, AUC 0.63'),
    ], className='rb-meta'),
], className='rb-header')

glossary = html.Div([
    *corners(),
    html.Div('How to read this page', className='rb-question'),
    html.Div([
        html.Div([html.B('Churn rate — '), 'the % of customers who canceled. Lower is better.']),
        html.Div([html.B('GRR — '), 'the % of revenue kept from existing customers, ignoring any new sales. 100% would mean nobody left.']),
        html.Div([html.B('Health score — '), 'a 0–100 ranking of each account relative to the others in this book (100 = safest). It is not a probability — it only says who looks riskier than whom.']),
    ], className='rb-glossary'),
], className='rb-card blueprint mb-5')

overview_tab = html.Div([
    html.Div('Revenue risk — active book', className='rb-section-title'),
    html.Div(
        f"{len(clients):,} active accounts · ${total_margin:,.0f} net margin · "
        "curated public dataset (BCG X / PowerCo), not operational data from any employer.",
        className='rb-section-note',
    ),
    glossary,
    kpi_row,

    dbc.Row([
        dbc.Col([
            html.Div('Where is our margin concentrated relative to risk?', className='rb-question'),
            dcc.Graph(figure=fig_decile, config={'displayModeBar': False}),
        ], md=6),
        dbc.Col([
            html.Div('How many accounts are in each risk tier?', className='rb-question'),
            dcc.Graph(figure=fig_bands, config={'displayModeBar': False}),
        ], md=6),
    ], className='mb-5 g-4'),

    dbc.Row([
        dbc.Col([
            html.Div('Risk queue — top 8 by predicted risk', className='rb-section-title', style={'fontSize': '16px'}),
            risk_queue_table,
        ], md=6),
        dbc.Col([
            html.Div('What the model says', className='rb-section-title', style={'fontSize': '16px'}),
            findings_list,
            html.Div('HEALTH SCORE = MODEL OUTPUT · CHURN AND GRR = ACTUALS',
                      style={'fontSize': '11px', 'color': 'var(--color-neutral-500)', 'marginTop': '12px', 'letterSpacing': '0.06em'}),
        ], md=6),
    ], className='mb-5 g-4'),
])

# ---------- recommended actions (5W2H) -- business judgment on top of the real findings above ----------
ACTION_CHANNEL = {
    'what': 'Audit sales qualification and first-90-day onboarding specific to Channel A.',
    'why': 'Channel A churns at 12.1% vs. 5.6% for Channel C — and Channel C also carries higher '
           'average margin ($238 vs. $199). Same product, worse fit at the channel level.',
    'who': 'Sales Ops (process audit) + Customer Success (onboarding redesign)',
    'when': 'Before the next acquisition push through this channel — target: next quarter',
    'where': 'All Channel A accounts, new and existing',
    'how': "Compare qualification criteria and first-90-day touchpoints between Channel A and Channel C; close the gap.",
    'how_much': '~443 accounts churn in excess of Channel C\'s rate — about $88,100/year in margin at stake.',
    'pdca': True,
}
ACTION_TENURE = {
    'what': 'Automated proactive renewal/health-check touchpoint triggered at month 30.',
    'why': 'Churn peaks in years 3–4 (12–14%) vs. 7.1% at year 6 — a specific early-life risk window, not a steady decline.',
    'who': 'Customer Success / Account Management',
    'when': 'Ongoing — triggered automatically as each account approaches its 3rd anniversary',
    'where': 'All accounts approaching the year-3 mark',
    'how': 'CS workflow trigger + a structured check-in call/review at month 30',
    'how_much': '$217,901 in margin already lost historically from the years 3–4 cohort — the size of the problem this targets.',
    'pdca': True,
}
ACTION_PRODUCTS = {
    'what': 'Correct internal messaging: stop presenting cross-sell as a churn-reduction lever.',
    'why': '1 product churns at 10.0%, 4+ products at 8.7% — too small a gap to call it a retention strategy.',
    'who': 'RevOps (KPI/metric definitions) + Sales enablement (playbook messaging)',
    'when': 'Next sales enablement content refresh',
    'where': 'Internal playbooks only — not a customer-facing change',
    'how': 'Reframe cross-sell as revenue expansion in CS/Sales materials, not as a retention play',
    'how_much': 'N/A — this is a correction, not an investment; it avoids spending retention effort on a lever the data doesn\'t support.',
    'pdca': False,
}
ACTION_FUEL = {
    'what': 'Targeted gas cross-sell campaign for electricity-only accounts, prioritized by risk score.',
    'why': 'Dual-fuel accounts churn less (8.2% vs. 10.1%) AND carry higher average margin ($249 vs. $176) — a margin upside, not just a retention story.',
    'who': 'Marketing (campaign) + Sales (offer/close) + Customer Success (targeting)',
    'when': 'Next campaign planning cycle',
    'where': '11,955 electricity-only accounts, highest-risk decile first',
    'how': 'Bundle offer for adding gas service; outreach list = electricity-only ∩ high risk',
    'how_much': '~$87,000/year in margin upside from the average per-account gap alone, before counting any churn reduction.',
    'pdca': True,
}
ACTION_CAMPAIGN = {
    'what': "Review Campaign A's targeting/qualification, or shift acquisition budget mix toward Campaign B/D-like profiles.",
    'why': 'Campaign A brings the most accounts (7,097) but the worst churn (12.6%); Campaign B brings fewer (4,294) at much better retention (6.0%).',
    'who': 'Marketing (campaign strategy) + Finance/RevOps (budget allocation)',
    'when': 'Next budget planning cycle',
    'where': 'Acquisition spend allocation across campaigns',
    'how': 'Build an LTV-adjusted return comparison per campaign before the next budget cycle — not just cost-per-acquisition',
    'how_much': "Not enough data here for a real number — this needs an LTV model (customer lifetime, not just this snapshot) before quoting a dollar figure. Flagged as a gap, not guessed.",
    'pdca': True,
}

# ---------- per-account recommended action (Account Detail tab) ----------
# Keyed off the #1 standout factor from notebooks/01_churn_model.py (a z-score
# comparison against the book average on the model's top features -- not SHAP,
# see the note in that file for why). Direction-aware where it changes the
# recommendation; a shared template otherwise. Rule-based on purpose: a lookup
# table is something Lucas can read and defend line by line, unlike a second
# model making the recommendation.
FACTOR_ACTION_TEMPLATES = {
    ('recent consumption drop', 'above'): {
        'what': 'Urgent usage-check call',
        'why': "This account's usage has dropped more sharply than most in the book — historically the single "
               "strongest churn signal here (see Overview).",
        'who': 'Customer Success', 'when': 'Within 2 weeks — this is the most time-sensitive signal available',
        'how': 'Confirm whether the drop is operational (e.g. seasonal, downsizing) or dissatisfaction-driven',
    },
    ('annual electricity consumption', 'below'): {
        'what': 'Usage check-in call',
        'why': "Usage is well below the book average.",
        'who': 'Customer Success', 'when': 'Before next renewal',
        'how': 'Understand whether usage is stabilizing at a lower level or still declining',
    },
    ('annual electricity consumption', 'above'): {
        'what': 'High-touch proactive outreach',
        'why': "Usage is well above the book average — a high-volume account.",
        'who': 'Account Management', 'when': 'Proactively, ahead of any renewal',
        'how': 'Losing an account this size would be a large single hit — treat as high-touch regardless of risk score alone',
    },
    ('tenure', 'below'): {
        'what': 'Early-tenure check-in',
        'why': 'This is a newer account, inside the higher-risk early-tenure window (years 3–4, see Drivers tab).',
        'who': 'Customer Success', 'when': 'Now, ahead of the year-3 mark',
        'how': 'Apply the standard early-tenure check-in motion (see the Drivers tab tenure action)',
    },
    ('tenure', 'above'): {
        'what': 'Personal outreach from the account owner',
        'why': 'This is a long-tenured account still showing up as high risk — loyalty alone is not protecting it.',
        'who': 'Account Management', 'when': 'Soon — this is not the expected profile for this risk level',
        'how': 'Investigate what changed recently rather than assuming tenure will hold',
    },
    ('price volatility (off-peak)', 'above'): {
        'what': 'Rate plan review',
        'why': "This account's price has moved around more than most, in the off-peak rate.",
        'who': 'Sales / Account Management', 'when': 'Next pricing review cycle',
        'how': 'Review the rate plan for stability; consider a fixed-rate offer',
    },
    ('price volatility (peak)', 'above'): {
        'what': 'Rate plan review',
        'why': "This account's peak-rate price has moved around more than most.",
        'who': 'Sales / Account Management', 'when': 'Next pricing review cycle',
        'how': 'Review the rate plan for stability; consider a fixed-rate offer',
    },
    ('average price level (off-peak)', 'above'): {
        'what': 'Rate review conversation',
        'why': 'Paying an above-average off-peak rate.',
        'who': 'Sales / Account Management', 'when': 'Next renewal conversation',
        'how': 'Get ahead of a competitor offering a lower rate',
    },
    ('net margin', 'above'): {
        'what': 'White-glove outreach',
        'why': 'This is a high-margin account — the financial stakes of losing it are larger than most.',
        'who': 'Account Management', 'when': 'Now',
        'how': 'Personal, non-automated outreach given what is at stake',
    },
    ('net margin', 'below'): {
        'what': 'Low-touch check-in',
        'why': 'This is a lower-margin account.',
        'who': 'Customer Success', 'when': 'Next scheduled cycle',
        'how': 'Email or self-serve check-in is proportionate here',
    },
    ('time since the last account change', 'above'): {
        'what': 'Proactive account review',
        'why': 'No changes on this account in longer than most — a possible disengagement signal.',
        'who': 'Customer Success', 'when': 'Next scheduled cycle',
        'how': 'Stale accounts rarely self-report a problem — check in before they raise one',
    },
    ('time since last renewal', 'above'): {
        'what': 'Renewal-readiness conversation',
        'why': 'Longer than average since the last renewal touchpoint.',
        'who': 'Customer Success / Account Management', 'when': 'Now',
        'how': 'Schedule a renewal-readiness conversation',
    },
    ('time to contract end', 'below'): {
        'what': 'Renewal conversation',
        'why': 'Contract end is closer than average.',
        'who': 'Customer Success / Account Management', 'when': 'Now, ahead of the contract date',
        'how': 'Standard renewal conversation, timed to the actual contract date',
    },
}
DEFAULT_ACTION_TEMPLATE = {
    'what': 'Account review',
    'why': 'This factor stands out relative to the book average.',
    'who': 'Customer Success', 'when': 'Next scheduled cycle',
    'how': 'Review to confirm whether this is a risk signal or a false positive',
}

def build_account_action(row):
    key = (row['factor_1_label'], row['factor_1_direction'])
    template = FACTOR_ACTION_TEMPLATES.get(key, DEFAULT_ACTION_TEMPLATE)
    why_all = ' · '.join(
        f"{row[f'factor_{i}_label']} ({row[f'factor_{i}_direction']} average)"
        for i in (1, 2, 3) if row[f'factor_{i}_label']
    )
    return {
        'what': template['what'],
        'why': f"{template['why']} Full picture for this account: {why_all}.",
        'who': template['who'],
        'when': template['when'],
        'where': 'This account only',
        'how': template['how'],
        'how_much': f"${row['net_margin']:,.0f} in net margin on this account.",
        'pdca': False,
    }

drivers_tab = html.Div([
    html.Div('What drives churn in this book?', className='rb-section-title'),
    html.Div(
        f"Five questions the data can actually answer. Groups with fewer than {MIN_N} accounts are left "
        "out of the charts below — too small a sample to read anything into.",
        className='rb-section-note',
    ),
    dbc.Row([
        dbc.Col(question_card(
            'Does the sales channel affect churn?',
            fig_channel,
            "Channel A brings in 46% of all accounts and churns at 12.1% — well above the 9.7% book "
            "average. Channel C (1,843 accounts) churns at just 5.6%, and also carries higher average "
            "margin. Same product, worse fit at the channel level.",
            action=ACTION_CHANNEL, card_id='action-channel',
        ), md=6),
        dbc.Col(question_card(
            'Does tenure protect against churn?',
            fig_tenure,
            "Mostly yes, but not right away: churn is highest in years 3–4 (12–14%) and drops to its "
            "lowest around year 6 (7.1%, the largest single cohort). The relationship has to survive "
            "the first few years before tenure starts working in the customer's favor.",
            action=ACTION_TENURE, card_id='action-tenure',
        ), md=6),
    ], className='g-4 mb-4'),
    dbc.Row([
        dbc.Col(question_card(
            'Do customers with more products churn less?',
            fig_products,
            "Barely any effect: 10.0% churn on 1 product vs. 8.7% on 4+. Cross-sell may still be worth "
            "doing for revenue, but this data doesn't support it as a retention lever on its own.",
            action=ACTION_PRODUCTS, card_id='action-products',
        ), md=6),
        dbc.Col(question_card(
            'Are dual-fuel customers stickier than single-service ones?',
            fig_fuel,
            "Yes, and it's not just retention: dual-fuel accounts churn less (8.2% vs. 10.1%) AND carry "
            "higher average margin ($249 vs. $176) than electricity-only accounts. Bundling a second "
            "service pays twice.",
            action=ACTION_FUEL, card_id='action-fuel',
        ), md=6),
    ], className='g-4 mb-4'),
    dbc.Row([
        dbc.Col(question_card(
            'Which acquisition channel brings in the most valuable customers — not just the most of them?',
            fig_campaign,
            "Campaign A brings the most accounts (7,097) at solid margin, but also the worst churn "
            "(12.6%). Campaign B brings fewer accounts at lower average margin, but keeps them far "
            "better (6.0% churn). Volume and quality are not the same channel here.",
            action=ACTION_CAMPAIGN, card_id='action-campaign',
        ), md=12),
    ], className='g-4 mb-4'),
])

ACTION_CARD_IDS = ['action-channel', 'action-tenure', 'action-products', 'action-fuel', 'action-campaign']

account_search = dbc.Input(
    id='account-search',
    type='text',
    placeholder='Search by account ID…',
    className='rb-search',
)

account_book_tab = html.Div([
    html.Div('Account book', className='rb-section-title'),
    html.Div(f"{len(clients):,} active accounts.", className='rb-section-note'),
    dbc.Row([
        dbc.Col(band_filter, md=6),
        dbc.Col(account_search, md=4),
    ], className='mb-3 align-items-center'),
    accounts_grid,
])

DEFAULT_DETAIL_ID = clients.sort_values('risk_rank').iloc[0]['id']  # #1 highest-risk account, shown until someone searches

account_detail_tab = html.Div([
    html.Div('Account detail', className='rb-section-title'),
    html.Div(
        "Why one specific account is flagged, and what to do about it -- not just the segment-level "
        "story from the Drivers tab. Paste an account ID from the Account Book, or leave it as-is to "
        "see the single highest-risk account in the book.",
        className='rb-section-note',
    ),
    dbc.Input(
        id='detail-account-input', type='text', value=DEFAULT_DETAIL_ID,
        placeholder='Paste an account ID…', className='rb-search', style={'maxWidth': '420px', 'marginBottom': '20px'},
    ),
    html.Div(id='account-detail-content'),
])

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(overview_tab, label='Overview', tab_id='overview'),
        dbc.Tab(drivers_tab, label='Drivers', tab_id='drivers'),
        dbc.Tab(account_book_tab, label='Account book', tab_id='book'),
        dbc.Tab(account_detail_tab, label='Account detail', tab_id='detail'),
    ], id='rb-tabs', active_tab='overview', className='rb-tabs mb-4'),
], fluid=True, style={'maxWidth': '1200px', 'padding': '32px 24px 60px'})


@app.callback(
    Output('accounts-grid', 'rowData'),
    Input('band-filter', 'value'),
    Input('account-search', 'value'),
)
def filter_grid(selected_band, search_text):
    filtered = clients
    if selected_band != 'All':
        filtered = filtered[filtered['health_band'] == selected_band]
    if search_text:
        filtered = filtered[filtered['id'].str.contains(search_text, case=False, na=False)]
    return filtered.sort_values('health_score').to_dict('records')


# one toggle callback per action panel -- five small callbacks, each trivial,
# rather than one clever pattern-matching callback for a fixed, known set of cards
for _cid in ACTION_CARD_IDS:
    app.callback(
        Output(f'{_cid}-collapse', 'is_open'),
        Input(f'{_cid}-toggle', 'n_clicks'),
        State(f'{_cid}-collapse', 'is_open'),
        prevent_initial_call=True,
    )(lambda n_clicks, is_open: not is_open)


@app.callback(
    Output('account-detail-content', 'children'),
    Input('detail-account-input', 'value'),
)
def render_account_detail(account_id):
    if not account_id:
        return html.Div("Paste an account ID above.", className='rb-kpi-note')
    match = clients[clients['id'] == account_id.strip()]
    if match.empty:
        return html.Div(
            f'No account found with ID "{account_id}". IDs are case-sensitive and come from the Account Book tab.',
            className='rb-kpi-note',
        )
    row = match.iloc[0]
    action = build_account_action(row)

    profile = dbc.Row([
        dbc.Col(kpi_card('Health score', f"{int(row['health_score'])}/100", row['health_band']), md=3),
        dbc.Col(kpi_card('Net margin', f"${row['net_margin']:,.0f}", f"Risk rank {int(row['risk_rank']):,} of {len(clients):,}"), md=3),
        dbc.Col(kpi_card('Tenure', f"{int(row['tenure_years'])} yrs", f"{int(row['nb_prod_act'])} product(s) contracted"), md=3),
        dbc.Col(kpi_card('Days to contract end', f"{int(row['days_to_contract_end']):,}", f"Churn probability (model, uncalibrated): {row['churn_probability']:.1%}"), md=3),
    ], className='g-3 mb-4')

    why_rows = [
        html.Div(f"{row[f'factor_{i}_label']} — {row[f'factor_{i}_direction']} the book average", className='rb-finding')
        for i in (1, 2, 3) if row[f'factor_{i}_label']
    ]
    why_panel = html.Div([
        *corners(),
        html.Div('Why this account is flagged', className='rb-question'),
        html.Div(
            "Not the model's internal logic (that would need SHAP, which this baseline doesn't use) — just "
            "where this account sits furthest from the book average on the features the model leans on most.",
            className='rb-kpi-note', style={'marginBottom': '8px'},
        ),
        *why_rows,
    ], className='rb-card blueprint mb-4')

    action_card = html.Div([
        *corners(),
        html.Div(f"Recommended action: {action['what']}", className='rb-question'),
        action_panel(action),
    ], className='rb-card blueprint')

    return html.Div([profile, why_panel, action_card])


if __name__ == '__main__':
    # local dev only -- in production (Render), gunicorn imports `server` above
    # directly and never runs this block
    debug = os.environ.get('DASH_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=debug, host='0.0.0.0', port=port)
