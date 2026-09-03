"""
Piece 1 - Customer Health & Churn Risk (PowerCo, SME/B2B energy)

Visual design ported from Lucas's Claude Design mockup (Desktop/Retention Book.html)
-- the "blueprint" system: Barlow/Barlow Condensed, navy/blue-gray tokens, sharp
corners, plus-sign corner marks. Numbers here are real, computed from
notebooks/01_churn_model.py -- the mockup used illustrative placeholder figures
(e.g. "12 accounts, $426k MRR") that do not appear anywhere in this file.

Two tabs: Overview (KPIs, charts, risk queue, findings) and Account Book
(searchable/filterable grid of all active accounts). A third tab in the mockup,
Account Detail (per-account drill-down), is not built yet.

Data: curated public sample (Kaggle mirror of the BCG X Data Science Job
Simulation dataset), not operational data from any employer. Risk score is an
illustrative baseline model (Random Forest, AUC 0.63) used only to rank accounts
by relative risk -- see notebooks/01_churn_model.py for the full method and its
limitations.
"""
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import Dash, dcc, html, Input, Output

CLEAN_DIR = '../data/clean'

clients = pd.read_csv(f'{CLEAN_DIR}/clients_scored.csv')
kpi = pd.read_csv(f'{CLEAN_DIR}/kpi_summary.csv').iloc[0]
churn_drivers = pd.read_csv(f'{CLEAN_DIR}/churn_drivers.csv')
campaign_value = pd.read_csv(f'{CLEAN_DIR}/campaign_value.csv')

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

def question_card(question, chart, answer):
    return html.Div([
        *corners(),
        html.Div(question, className='rb-question'),
        dcc.Graph(figure=chart, config={'displayModeBar': False}),
        html.Div(answer, className='rb-kpi-note', style={'marginTop': '4px'}),
    ], className='rb-card blueprint', style={'marginBottom': '8px'})

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
            "average. Channel C (1,843 accounts) churns at just 5.6%. Same product, different result "
            "depending on how the customer was sold to.",
        ), md=6),
        dbc.Col(question_card(
            'Does tenure protect against churn?',
            fig_tenure,
            "Mostly yes, but not right away: churn is highest in years 3–4 (12–14%) and drops to its "
            "lowest around year 6 (7.1%, the largest single cohort). The relationship has to survive "
            "the first few years before tenure starts working in the customer's favor.",
        ), md=6),
    ], className='g-4 mb-4'),
    dbc.Row([
        dbc.Col(question_card(
            'Do customers with more products churn less?',
            fig_products,
            "Barely any effect: 10.0% churn on 1 product vs. 8.7% on 4+. Cross-sell may still be worth "
            "doing for revenue, but this data doesn't support it as a retention lever on its own.",
        ), md=6),
        dbc.Col(question_card(
            'Are dual-fuel customers stickier than single-service ones?',
            fig_fuel,
            "Yes, modestly: 8.2% churn for electricity + gas customers vs. 10.1% for electricity only. "
            "Bundling a second service correlates with staying longer.",
        ), md=6),
    ], className='g-4 mb-4'),
    dbc.Row([
        dbc.Col(question_card(
            'Which acquisition channel brings in the most valuable customers — not just the most of them?',
            fig_campaign,
            "Campaign A brings the most accounts (7,097) at solid margin, but also the worst churn "
            "(12.6%). Campaign B brings fewer accounts at lower average margin, but keeps them far "
            "better (6.0% churn). Volume and quality are not the same channel here.",
        ), md=12),
    ], className='g-4 mb-4'),
])

account_book_tab = html.Div([
    html.Div('Account book', className='rb-section-title'),
    html.Div(f"{len(clients):,} active accounts.", className='rb-section-note'),
    dbc.Row([
        dbc.Col(band_filter, md=8),
    ], className='mb-3'),
    accounts_grid,
])

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(overview_tab, label='Overview', tab_id='overview'),
        dbc.Tab(drivers_tab, label='Drivers', tab_id='drivers'),
        dbc.Tab(account_book_tab, label='Account book', tab_id='book'),
    ], id='rb-tabs', active_tab='overview', className='rb-tabs mb-4'),
], fluid=True, style={'maxWidth': '1200px', 'padding': '32px 24px 60px'})


@app.callback(
    Output('accounts-grid', 'rowData'),
    Input('band-filter', 'value'),
)
def filter_grid_by_band(selected_band):
    if selected_band == 'All':
        filtered = clients
    else:
        filtered = clients[clients['health_band'] == selected_band]
    return filtered.sort_values('health_score').to_dict('records')


if __name__ == '__main__':
    app.run(debug=True)
