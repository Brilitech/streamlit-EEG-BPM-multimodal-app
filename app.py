import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Focus Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE - THEME
# ============================================================================

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEMES = {
    'dark': {
        'bg-primary': '#050810',
        'bg-secondary': '#0d1117',
        'bg-card': '#111827',
        'accent-green': '#00e5a0',
        'accent-red': '#ff3d5a',
        'accent-blue': '#38bdf8',
        'accent-yellow': '#fbbf24',
        'text-primary': '#f0f4ff',
        'text-secondary': '#6b7fa3',
        'border': '#1e2d45',
        'bg-chart': 'rgba(11,17,27,1)',
        'plot-chart': 'rgba(11,17,27,1)',
        'grid-color': '#1e2d45',
        'is_dark': True,
        'rgb-green': '0,229,160',
        'rgb-red': '255,61,90',
        'rgb-blue': '56,189,248',
    },
    'light': {
        'bg-primary': '#f0f4f8',
        'bg-secondary': '#ffffff',
        'bg-card': '#ffffff',
        'accent-green': '#059669',
        'accent-red': '#dc2626',
        'accent-blue': '#0284c7',
        'accent-yellow': '#d97706',
        'text-primary': '#0f172a',
        'text-secondary': '#64748b',
        'border': '#e2e8f0',
        'bg-chart': 'rgba(255,255,255,1)',
        'plot-chart': 'rgba(255,255,255,1)',
        'grid-color': '#e2e8f0',
        'is_dark': False,
        'rgb-green': '5,150,105',
        'rgb-red': '220,38,38',
        'rgb-blue': '2,132,199',
    }
}

current_theme = THEMES[st.session_state.theme_mode]
is_dark = current_theme['is_dark']

# ============================================================================
# CUSTOM CSS
# ============================================================================

def get_css(t):
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    :root {{
        --bg-primary:    {t['bg-primary']};
        --bg-secondary:  {t['bg-secondary']};
        --bg-card:       {t['bg-card']};
        --accent-green:  {t['accent-green']};
        --accent-red:    {t['accent-red']};
        --accent-blue:   {t['accent-blue']};
        --accent-yellow: {t['accent-yellow']};
        --text-primary:  {t['text-primary']};
        --text-secondary:{t['text-secondary']};
        --border:        {t['border']};
        --rgb-green:     {t['rgb-green']};
        --rgb-red:       {t['rgb-red']};
        --rgb-blue:      {t['rgb-blue']};
    }}

    /* ── Base ── */
    .stApp {{
        background: var(--bg-primary);
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--text-primary);
    }}
    #MainMenu, footer, .stDeployButton {{ visibility: hidden; display: none; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stSidebar"] * {{ color: var(--text-primary) !important; }}

    /* ── Header Banner ── */
    .main-header {{
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
        border: 1px solid var(--border);
        border-top: 3px solid var(--accent-green);
        border-radius: 0 0 16px 16px;
        padding: 36px 44px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .main-header::after {{
        content: '';
        position: absolute;
        right: 40px;
        top: 50%;
        transform: translateY(-50%);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: rgba(var(--rgb-green), 0.04);
        letter-spacing: 4px;
        white-space: nowrap;
        pointer-events: none;
    }}
    .header-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 3px;
        color: var(--accent-green);
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .header-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}
    .header-sub {{
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin: 0;
    }}
    .header-badges {{
        margin-top: 18px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.5px;
        font-weight: 500;
    }}
    .badge-green {{
        background: rgba(var(--rgb-green), 0.1);
        border: 1px solid rgba(var(--rgb-green), 0.3);
        color: var(--accent-green);
    }}
    .badge-blue {{
        background: rgba(var(--rgb-blue), 0.1);
        border: 1px solid rgba(var(--rgb-blue), 0.3);
        color: var(--accent-blue);
    }}

    /* ── Metric Cards ── */
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .metric-card:hover {{
        border-color: rgba(var(--rgb-green), 0.4);
        box-shadow: 0 0 20px rgba(var(--rgb-green), 0.06);
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
    }}
    .mc-green::before  {{ background: var(--accent-green); }}
    .mc-blue::before   {{ background: var(--accent-blue); }}
    .mc-red::before    {{ background: var(--accent-red); }}
    .mc-yellow::before {{ background: var(--accent-yellow); }}

    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }}
    .metric-sub {{
        font-size: 0.77rem;
        color: var(--text-secondary);
        margin-top: 7px;
    }}

    /* ── Verdict ── */
    .verdict-wrap {{
        border-radius: 12px;
        padding: 30px 36px;
        text-align: center;
        border: 1px solid var(--border);
        position: relative;
        overflow: hidden;
    }}
    .verdict-fokus {{
        background: linear-gradient(135deg,
            rgba(var(--rgb-green), 0.08) 0%,
            rgba(var(--rgb-green), 0.03) 100%);
        border-color: rgba(var(--rgb-green), 0.35);
    }}
    .verdict-tidak {{
        background: linear-gradient(135deg,
            rgba(var(--rgb-red), 0.08) 0%,
            rgba(var(--rgb-red), 0.03) 100%);
        border-color: rgba(var(--rgb-red), 0.35);
    }}
    .verdict-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }}
    .verdict-main {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1;
    }}
    .verdict-green {{ color: var(--accent-green); }}
    .verdict-red   {{ color: var(--accent-red); }}
    .verdict-meta {{
        font-size: 0.83rem;
        color: var(--text-secondary);
        margin-top: 10px;
    }}

    /* ── Section Label ── */
    .section-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--text-secondary);
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-label::before {{
        content: '';
        width: 14px;
        height: 2px;
        background: var(--accent-green);
        display: inline-block;
    }}

    /* ── Info Box ── */
    .info-box {{
        background: rgba(var(--rgb-blue), 0.07);
        border: 1px solid rgba(var(--rgb-blue), 0.22);
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin: 14px 0;
        line-height: 1.7;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: var(--accent-green);
        color: #050810;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 1.5px;
        padding: 11px 28px;
        width: 100%;
        text-transform: uppercase;
        transition: opacity 0.15s, transform 0.1s;
    }}
    .stButton > button:hover {{
        opacity: 0.88;
        transform: translateY(-1px);
    }}
    .stButton > button:disabled {{
        background: var(--border);
        color: var(--text-secondary);
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 1px;
        color: var(--text-secondary);
        border-radius: 5px;
        padding: 8px 20px;
        text-transform: uppercase;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(var(--rgb-green), 0.12) !important;
        color: var(--accent-green) !important;
    }}

    /* ── Spinner ── */
    .stSpinner > div {{
        border-top-color: var(--accent-green) !important;
    }}

    /* ── Sidebar logo area ── */
    .sidebar-logo {{
        text-align: center;
        padding: 20px 0 28px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
    }}
    .sidebar-logo-text {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--accent-green);
        letter-spacing: -0.5px;
    }}
    .sidebar-logo-sub {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-secondary);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }}

    /* ── Model status card ── */
    .model-status {{
        background: rgba(var(--rgb-green), 0.06);
        border: 1px solid rgba(var(--rgb-green), 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 0.79rem;
        line-height: 1.9;
    }}
    .model-status-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        color: var(--accent-green);
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .model-status-title::before {{
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-green);
        display: inline-block;
        box-shadow: 0 0 6px var(--accent-green);
    }}
    .ms-key {{ color: var(--text-secondary); }}
    .ms-val {{ color: var(--text-primary); font-weight: 500; }}

    /* ── Upload section header ── */
    .sidebar-section-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin: 20px 0 10px;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ── Step cards (landing) ── */
    .step-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 28px 24px;
        text-align: center;
        transition: border-color 0.2s, transform 0.2s;
    }}
    .step-card:hover {{
        border-color: rgba(var(--rgb-green), 0.3);
        transform: translateY(-2px);
    }}
    .step-num {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: rgba(var(--rgb-green), 0.25);
        line-height: 1;
        margin-bottom: 12px;
    }}
    .step-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    .step-desc {{
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }}
</style>
"""

st.markdown(get_css(current_theme), unsafe_allow_html=True)

# ============================================================================
# LSTM MODEL
# ============================================================================

class FocusLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(FocusLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    scaler     = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    pca        = joblib.load(os.path.join(models_dir, 'pca.pkl'))
    le         = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    data_info  = joblib.load(os.path.join(models_dir, 'data_info.pkl'))
    checkpoint = torch.load(os.path.join(models_dir, 'best_model.pth'), map_location='cpu')

    cfg = checkpoint.get('model_config', {
        'input_size': 6, 'hidden_size': 64,
        'num_layers': 2, 'num_classes': 2, 'dropout': 0.3
    })
    model = FocusLSTM(
        input_size=cfg['input_size'], hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'], num_classes=cfg['num_classes'],
        dropout=cfg.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, scaler, pca, le, data_info, checkpoint

# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(eeg_df_raw, bpm_df_raw, model, scaler, pca, le, data_info):
    EEG_COLS     = ['Timestamp', 'Low Alpha', 'High Alpha', 'Low Beta', 'High Beta']
    BPM_COLS     = ['Timestamp', 'BPM', 'Avg BPM']
    FEATURE_COLS = ['Low Alpha', 'High Alpha', 'Low Beta', 'High Beta', 'BPM', 'Avg BPM']

    eeg_df = eeg_df_raw[[c for c in EEG_COLS if c in eeg_df_raw.columns]].copy()
    bpm_df = bpm_df_raw[[c for c in BPM_COLS if c in bpm_df_raw.columns]].copy()

    eeg_df['ts_key'] = eeg_df['Timestamp'].astype(str).str.replace('"', '').str.replace(',', ' ').str[:19]
    bpm_df['ts_key'] = bpm_df['Timestamp'].astype(str).str[:19]

    bpm_clean = bpm_df.drop_duplicates(subset=['ts_key'])
    merged    = eeg_df.merge(bpm_clean, on='ts_key', how='inner', suffixes=('', '_bpm'))

    if merged.empty:
        return None, "No matching timestamps between EEG and BPM!"

    for col in FEATURE_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('float32')

    data_clean  = merged[FEATURE_COLS].dropna()
    data_scaled = scaler.transform(data_clean.values)
    data_pca    = pca.transform(data_scaled)

    window_size = data_info.get('window_size', 30)
    windows = [data_pca[i:i+window_size]
               for i in range(0, len(data_pca) - window_size + 1, window_size)]

    if not windows:
        return None, f"Insufficient data! Need at least {window_size} samples, got {len(data_pca)}."

    X_input = torch.tensor(np.array(windows), dtype=torch.float32)

    with torch.no_grad():
        outputs       = model(X_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted  = torch.max(outputs, 1)

    predictions = le.inverse_transform(predicted.numpy())
    probs       = probabilities.numpy()

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        pred_idx = predicted[i].item()
        fokus_idx = le.transform(['Fokus'])[0] if 'Fokus' in le.classes_ else 0
        tidak_idx = le.transform(['Tidak Fokus'])[0] if 'Tidak Fokus' in le.classes_ else 1
        results.append({
            'Window': i + 1,
            'Start (s)': round(i * 0.3, 1),
            'End (s)': round((i + 1) * 0.3, 1),
            'Prediction': pred,
            'Confidence (%)': round(prob[pred_idx] * 100, 2),
            'Fokus Prob (%)': round(prob[fokus_idx] * 100, 2),
            'Tidak Fokus Prob (%)': round(prob[tidak_idx] * 100, 2),
        })

    return pd.DataFrame(results), None

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def chart_timeline(df, t):
    colors = [t['accent-green'] if p == 'Fokus' else t['accent-red'] for p in df['Prediction']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Window'], y=df['Confidence (%)'],
        marker_color=colors, marker_line_width=0,
        hovertemplate='<b>Window %{x}</b><br>%{customdata}<br>Confidence: %{y:.1f}%<extra></extra>',
        customdata=df['Prediction'],
    ))
    fig.update_layout(
        paper_bgcolor=t['bg-chart'], plot_bgcolor=t['plot-chart'],
        font=dict(family='IBM Plex Mono, monospace', color=t['text-secondary'], size=11),
        title=dict(text='Per-Window Prediction Timeline', font=dict(size=12, color=t['text-primary'])),
        xaxis=dict(title='Window', gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        yaxis=dict(title='Confidence (%)', range=[0, 108], gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=44, b=40),
        showlegend=False, height=300,
    )
    return fig

def chart_probability_line(df, t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Window'], y=df['Fokus Prob (%)'],
        name='Fokus', line=dict(color=t['accent-green'], width=2),
        fill='tozeroy', fillcolor=f"rgba({t['rgb-green']},0.09)",
        hovertemplate='Window %{x}<br>Fokus: %{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['Window'], y=df['Tidak Fokus Prob (%)'],
        name='Tidak Fokus', line=dict(color=t['accent-red'], width=2),
        fill='tozeroy', fillcolor=f"rgba({t['rgb-red']},0.09)",
        hovertemplate='Window %{x}<br>Tidak Fokus: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dot', line_color=t['text-secondary'], line_width=1)
    fig.update_layout(
        paper_bgcolor=t['bg-chart'], plot_bgcolor=t['plot-chart'],
        font=dict(family='IBM Plex Mono, monospace', color=t['text-secondary'], size=11),
        title=dict(text='Probability Over Time', font=dict(size=12, color=t['text-primary'])),
        xaxis=dict(title='Window', gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        yaxis=dict(title='Probability (%)', range=[0, 108], gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=44, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=t['border'],
                    font=dict(color=t['text-secondary'], size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=300,
    )
    return fig

def chart_donut(fokus_pct, tidak_pct, t):
    max_pct = max(fokus_pct, tidak_pct)
    fig = go.Figure(go.Pie(
        labels=['Fokus', 'Tidak Fokus'],
        values=[fokus_pct, tidak_pct],
        hole=0.68,
        marker=dict(colors=[t['accent-green'], t['accent-red']], line=dict(width=0)),
        textinfo='percent',
        textfont=dict(family='IBM Plex Mono', size=11, color=t['text-primary']),
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor=t['bg-chart'], plot_bgcolor=t['plot-chart'],
        font=dict(family='IBM Plex Mono, monospace', color=t['text-secondary'], size=11),
        title=dict(text='Distribution', font=dict(size=12, color=t['text-primary'])),
        margin=dict(l=30, r=30, t=44, b=30),
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=t['text-secondary'], size=10),
                    orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        height=300,
        annotations=[dict(
            text=f"<b>{max_pct:.0f}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family='IBM Plex Mono', size=20, color=t['text-primary'])
        )]
    )
    return fig

def chart_confidence_hist(df, t):
    fokus_conf = df[df['Prediction'] == 'Fokus']['Confidence (%)']
    tidak_conf = df[df['Prediction'] == 'Tidak Fokus']['Confidence (%)']
    fig = go.Figure()
    if len(fokus_conf):
        fig.add_trace(go.Histogram(
            x=fokus_conf, name='Fokus',
            marker_color=f"rgba({t['rgb-green']},0.7)",
            nbinsx=20, hovertemplate='Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))
    if len(tidak_conf):
        fig.add_trace(go.Histogram(
            x=tidak_conf, name='Tidak Fokus',
            marker_color=f"rgba({t['rgb-red']},0.7)",
            nbinsx=20, hovertemplate='Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))
    fig.update_layout(
        paper_bgcolor=t['bg-chart'], plot_bgcolor=t['plot-chart'],
        font=dict(family='IBM Plex Mono, monospace', color=t['text-secondary'], size=11),
        title=dict(text='Confidence Distribution', font=dict(size=12, color=t['text-primary'])),
        xaxis=dict(title='Confidence (%)', gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        yaxis=dict(title='Count', gridcolor=t['grid-color'], linecolor=t['border'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=44, b=40),
        barmode='overlay',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=t['text-secondary'], size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=300,
    )
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    # ── Theme toggle ──
    col_dark, col_light = st.columns(2, gap="small")
    with col_dark:
        if st.button("🌙 Dark", key="theme_dark", use_container_width=True):
            st.session_state.theme_mode = 'dark'
            st.rerun()
    with col_light:
        if st.button("☀️ Light", key="theme_light", use_container_width=True):
            st.session_state.theme_mode = 'light'
            st.rerun()

    # ── Logo ──
    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">🧠 FocusNet</div>
        <div class="sidebar-logo-sub">LSTM · EEG + BPM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ──
    try:
        model, scaler, pca, le, data_info, checkpoint = load_artifacts()
        st.markdown(f"""
        <div class="model-status">
            <div class="model-status-title">Model Loaded</div>
            <div><span class="ms-key">Epoch&nbsp;&nbsp;&nbsp; </span><span class="ms-val">{checkpoint.get('epoch', 0) + 1}</span></div>
            <div><span class="ms-key">Accuracy </span><span class="ms-val">{checkpoint.get('test_acc', 0):.2f}%</span></div>
            <div><span class="ms-key">Classes&nbsp; </span><span class="ms-val">{', '.join(le.classes_)}</span></div>
        </div>
        """, unsafe_allow_html=True)
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Model not found!\n\nLetakkan file model di folder `/models/`\n\n`{e}`")
        model_loaded = False

    # ── Upload ──
    st.markdown(f"""
    <div class="sidebar-section-title">── Upload Data ──</div>
    """, unsafe_allow_html=True)

    eeg_file = st.file_uploader("📊 EEG File (.csv)", type=['csv'], key="eeg")
    bpm_file = st.file_uploader("💓 BPM File (.csv)", type=['csv'], key="bpm")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶ RUN ANALYSIS", disabled=(not model_loaded), use_container_width=True)

    # ── Info ──
    st.markdown(f"""
    <div style="margin-top:24px; font-family:'IBM Plex Mono',monospace;
                font-size:0.63rem; color:{current_theme['text-secondary']}; line-height:2;">
        <div style="color:{current_theme['text-secondary']}; letter-spacing:1.5px;
                    margin-bottom:6px; text-transform:uppercase;">── Spesifikasi ──</div>
        <div>Window &nbsp;&nbsp;&nbsp;: 30 timesteps</div>
        <div>Durasi &nbsp;&nbsp;&nbsp;: 0.3s / window</div>
        <div>Features &nbsp;: 6 (post-PCA)</div>
        <div>Model &nbsp;&nbsp;&nbsp;&nbsp;: LSTM 2-layer</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <div class="header-eyebrow">Focus Detection System</div>
    <h1 class="header-title">🧠 Identifikasi Tingkat Fokus Mahasiswa berbasis Multimodal</h1>
    <p class="header-sub">EEG + BPM Multimodal Analysis &nbsp;·&nbsp; LSTM Deep Learning &nbsp;·&nbsp; Muhammad Azril Haidar Al Matiin — 23051640011</p>
    <div class="header-badges">
        <span class="badge badge-green">● Multimodal</span>
        <span class="badge badge-green">● TESIS</span>
        <span class="badge badge-blue">▲ v2.1</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEFAULT STATE
# ============================================================================

if not run_btn or eeg_file is None or bpm_file is None:

    if not model_loaded:
        st.warning("⚠️ Model files not found. Please check the `/models/` directory.")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">📊 Upload Data</div>
            <div class="step-desc">Upload file EEG CSV dan BPM CSV melalui sidebar di sebelah kiri</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">▶ Jalankan</div>
            <div class="step-desc">Klik tombol RUN ANALYSIS untuk memulai proses prediksi</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">📈 Lihat Hasil</div>
            <div class="step-desc">Analisis dashboard prediksi, grafik visualisasi, dan statistik</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top:24px">
        <b style="color:{current_theme['accent-blue']}">📌 Format File yang Dibutuhkan</b><br><br>
        <b>EEG CSV :</b> Timestamp, Low Alpha, High Alpha, Low Beta, High Beta<br>
        <b>BPM CSV :</b> Timestamp, BPM, Avg BPM
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================================
# INFERENCE
# ============================================================================

with st.spinner("🔄 Processing data and running inference..."):
    try:
        eeg_df_raw = pd.read_csv(eeg_file)
        bpm_df_raw = pd.read_csv(bpm_file)
        results_df, error = run_inference(eeg_df_raw, bpm_df_raw, model, scaler, pca, le, data_info)
        if error:
            st.error(f"❌ {error}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ============================================================================
# STATS
# ============================================================================

total_windows  = len(results_df)
total_duration = total_windows * 0.3
fokus_count    = (results_df['Prediction'] == 'Fokus').sum()
tidak_count    = (results_df['Prediction'] == 'Tidak Fokus').sum()
fokus_pct      = (fokus_count / total_windows) * 100
tidak_pct      = (tidak_count / total_windows) * 100
avg_conf       = results_df['Confidence (%)'].mean()
majority       = 'Fokus' if fokus_count >= tidak_count else 'Tidak Fokus'

# ============================================================================
# VERDICT
# ============================================================================

if majority == 'Fokus':
    st.markdown(f"""
    <div class="verdict-wrap verdict-fokus">
        <div class="verdict-tag">Overall Verdict</div>
        <div class="verdict-main verdict-green">🟢 FOKUS</div>
        <div class="verdict-meta">
            Dominan pada {fokus_pct:.1f}% sesi &nbsp;·&nbsp; Rata-rata confidence {avg_conf:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-wrap verdict-tidak">
        <div class="verdict-tag">Overall Verdict</div>
        <div class="verdict-main verdict-red">🔴 TIDAK FOKUS</div>
        <div class="verdict-meta">
            Dominan pada {tidak_pct:.1f}% sesi &nbsp;·&nbsp; Rata-rata confidence {avg_conf:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ============================================================================
# METRIC CARDS
# ============================================================================

c1, c2, c3, c4, c5 = st.columns(5, gap="small")

with c1:
    st.markdown(f"""
    <div class="metric-card mc-blue">
        <div class="metric-label">Total Windows</div>
        <div class="metric-value">{total_windows}</div>
        <div class="metric-sub">windows analyzed</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card mc-yellow">
        <div class="metric-label">Total Durasi</div>
        <div class="metric-value">{total_duration:.1f}s</div>
        <div class="metric-sub">{total_duration/60:.2f} menit</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card mc-green">
        <div class="metric-label">Fokus</div>
        <div class="metric-value" style="color:{current_theme['accent-green']}">{fokus_pct:.1f}%</div>
        <div class="metric-sub">{fokus_count} windows · {fokus_count*0.3:.1f}s</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card mc-red">
        <div class="metric-label">Tidak Fokus</div>
        <div class="metric-value" style="color:{current_theme['accent-red']}">{tidak_pct:.1f}%</div>
        <div class="metric-sub">{tidak_count} windows · {tidak_count*0.3:.1f}s</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card mc-blue">
        <div class="metric-label">Avg Confidence</div>
        <div class="metric-value">{avg_conf:.1f}%</div>
        <div class="metric-sub">model certainty</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ============================================================================
# CHARTS
# ============================================================================

st.markdown('<div class="section-label">Visualisasi</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1], gap="medium")
with col_left:
    st.plotly_chart(chart_timeline(results_df, current_theme), use_container_width=True)
with col_right:
    st.plotly_chart(chart_donut(fokus_pct, tidak_pct, current_theme), use_container_width=True)

col_a, col_b = st.columns(2, gap="medium")
with col_a:
    st.plotly_chart(chart_probability_line(results_df, current_theme), use_container_width=True)
with col_b:
    st.plotly_chart(chart_confidence_hist(results_df, current_theme), use_container_width=True)

# ============================================================================
# TABLE + EXPORT
# ============================================================================

st.markdown('<div class="section-label">Detail Data</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 PER-WINDOW RESULTS", "⬇️ EXPORT"])

with tab1:
    def color_prediction(val):
        if val == 'Fokus':
            return f'color: {current_theme["accent-green"]}; font-weight: bold'
        return f'color: {current_theme["accent-red"]}; font-weight: bold'

    styled_df = results_df.style.map(
        color_prediction, subset=['Prediction']
    ).format({
        'Confidence (%)': '{:.2f}',
        'Fokus Prob (%)': '{:.2f}',
        'Tidak Fokus Prob (%)': '{:.2f}',
        'Start (s)': '{:.1f}',
        'End (s)': '{:.1f}',
    }).set_properties(**{
        'background-color': current_theme['bg-card'],
        'color': current_theme['text-primary'],
        'border-color': current_theme['border'],
        'font-family': 'IBM Plex Mono, monospace',
        'font-size': '12px'
    })

    st.dataframe(styled_df, use_container_width=True, height=350)

with tab2:
    st.markdown("""
    <div class="info-box">
        Download hasil prediksi sebagai CSV untuk dokumentasi tesis.
    </div>
    """, unsafe_allow_html=True)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ DOWNLOAD RESULTS (.csv)",
        data=csv,
        file_name="focus_prediction_results.csv",
        mime="text/csv"
    )
