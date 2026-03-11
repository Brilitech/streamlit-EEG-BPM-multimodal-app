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
    st.session_state.theme_mode = 'dark'  # Default: dark

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEMES = {
    'dark': {
        'bg-primary': '#0a0e1a',
        'bg-secondary': '#111827',
        'bg-card': '#1a2035',
        'accent-green': '#00ff88',
        'accent-red': '#ff4757',
        'accent-blue': '#4facfe',
        'accent-yellow': '#ffd32a',
        'text-primary': '#e8eaf6',
        'text-secondary': '#8892b0',
        'border': '#2d3561',
        'bg-chart': 'rgba(26,32,53,1)',
        'plot-chart': 'rgba(26,32,53,1)',
        'grid-color': '#2d3561',
    },
    'light': {
        'bg-primary': '#f8f9fa',
        'bg-secondary': '#ffffff',
        'bg-card': '#ffffff',
        'accent-green': '#00a85c',
        'accent-red': '#d63447',
        'accent-blue': '#0066cc',
        'accent-yellow': '#ff9500',
        'text-primary': '#1a1a1a',
        'text-secondary': '#666666',
        'border': '#e0e0e0',
        'bg-chart': 'rgba(255,255,255,1)',
        'plot-chart': 'rgba(255,255,255,1)',
        'grid-color': '#e8e8e8',
    }
}

current_theme = THEMES[st.session_state.theme_mode]

# ============================================================================
# CUSTOM CSS - Dynamic Theme
# ============================================================================

def get_css(theme, is_dark):
    """Generate CSS based on current theme"""
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    
    :root {{
        --bg-primary: {theme['bg-primary']};
        --bg-secondary: {theme['bg-secondary']};
        --bg-card: {theme['bg-card']};
        --accent-green: {theme['accent-green']};
        --accent-red: {theme['accent-red']};
        --accent-blue: {theme['accent-blue']};
        --accent-yellow: {theme['accent-yellow']};
        --text-primary: {theme['text-primary']};
        --text-secondary: {theme['text-secondary']};
        --border: {theme['border']};
    }}
    
    /* Main background */
    .stApp {{
        background: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
        transition: background 0.3s ease, color 0.3s ease;
    }}
    
    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {theme['bg-primary']} 0%, {theme['bg-card']} 50%, {theme['bg-secondary']} 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba({'0,255,136' if is_dark else '0,168,92'}},0.06) 0%, transparent 70%);
        border-radius: 50%;
    }}
    
    .main-header::after {{
        content: '';
        position: absolute;
        bottom: -50%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba({'79,172,254' if is_dark else '0,102,204'}},0.06) 0%, transparent 70%);
        border-radius: 50%;
    }}
    
    .header-title {{
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, {theme['accent-green']}, {theme['accent-blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
    }}
    
    .header-subtitle {{
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 300;
        letter-spacing: 0.5px;
        transition: color 0.3s ease;
    }}
    
    .header-badge {{
        display: inline-block;
        background: rgba({'0,255,136' if is_dark else '0,168,92'},0.1);
        border: 1px solid rgba({'0,255,136' if is_dark else '0,168,92'},0.3);
        color: {theme['accent-green']};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin-top: 12px;
        letter-spacing: 1px;
        transition: background 0.3s ease, border-color 0.3s ease, color 0.3s ease;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
        box-shadow: {'0 2px 4px rgba(0,0,0,0.1)' if not is_dark else '0 2px 8px rgba(0,0,0,0.2)'};
    }}
    
    .metric-card:hover {{
        border-color: rgba({'0,255,136' if is_dark else '0,168,92'},0.3);
    }}
    
    .metric-label {{
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'Space Mono', monospace;
        margin-bottom: 8px;
        transition: color 0.3s ease;
    }}
    
    .metric-value {{
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        transition: color 0.3s ease;
    }}
    
    .metric-sub {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 6px;
        transition: color 0.3s ease;
    }}
    
    .metric-accent-green {{ border-left: 3px solid {theme['accent-green']}; }}
    .metric-accent-blue  {{ border-left: 3px solid {theme['accent-blue']}; }}
    .metric-accent-red   {{ border-left: 3px solid {theme['accent-red']}; }}
    .metric-accent-yellow{{ border-left: 3px solid {theme['accent-yellow']}; }}
    
    /* Verdict card */
    .verdict-fokus {{
        background: linear-gradient(135deg, rgba({'0,255,136' if is_dark else '0,168,92'},0.1) 0%, rgba({'0,255,136' if is_dark else '0,168,92'},0.05) 100%);
        border: 2px solid rgba({'0,255,136' if is_dark else '0,168,92'},0.4);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    .verdict-tidak-fokus {{
        background: linear-gradient(135deg, rgba({'255,71,87' if is_dark else '214,52,71'},0.1) 0%, rgba({'255,71,87' if is_dark else '214,52,71'},0.05) 100%);
        border: 2px solid rgba({'255,71,87' if is_dark else '214,52,71'},0.4);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    .verdict-label {{
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 8px;
        transition: color 0.3s ease;
    }}
    
    .verdict-text-fokus {{
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: {theme['accent-green']};
        transition: color 0.3s ease;
    }}
    
    .verdict-text-tidak-fokus {{
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: {theme['accent-red']};
        transition: color 0.3s ease;
    }}
    
    /* Section header */
    .section-header {{
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
        margin-bottom: 16px;
        transition: border-color 0.3s ease, color 0.3s ease;
    }}
    
    /* Upload area */
    .upload-container {{
        background: var(--bg-card);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }}
    
    /* Button */
    .stButton > button {{
        background: linear-gradient(135deg, {theme['accent-green']}, {theme['accent-blue']});
        color: {theme['bg-primary']};
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 1px;
        padding: 12px 32px;
        width: 100%;
        transition: opacity 0.2s ease, background 0.3s ease;
    }}
    
    .stButton > button:hover {{
        opacity: 0.85;
    }}

    /* Info box */
    .info-box {{
        background: rgba({'79,172,254' if is_dark else '0,102,204'},0.08);
        border: 1px solid rgba({'79,172,254' if is_dark else '0,102,204'},0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 12px 0;
        transition: background 0.3s ease, border-color 0.3s ease, color 0.3s ease;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {theme['accent-green']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg-card);
        border-radius: 8px;
        border: 1px solid var(--border);
        padding: 4px;
        gap: 4px;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
        color: var(--text-secondary);
        border-radius: 6px;
        padding: 8px 20px;
        transition: color 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba({'0,255,136' if is_dark else '0,168,92'},0.1) !important;
        color: {theme['accent-green']} !important;
    }}
    
    /* Theme toggle styling */
    .theme-toggle-btn {{
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 8px 16px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }}
    
    .theme-toggle-btn:hover {{
        border-color: {theme['accent-green']};
        color: {theme['accent-green']};
    }}
    
    .theme-toggle-btn.active {{
        background: rgba({'0,255,136' if is_dark else '0,168,92'},0.15);
        border-color: {theme['accent-green']};
        color: {theme['accent-green']};
    }}
</style>
"""

st.markdown(get_css(current_theme, st.session_state.theme_mode == 'dark'), unsafe_allow_html=True)

# ============================================================================
# LSTM MODEL DEFINITION
# ============================================================================

class FocusLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(FocusLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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
        out = self.fc(out)
        return out

# ============================================================================
# LOAD ARTIFACTS (CACHED)
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    scaler      = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    pca         = joblib.load(os.path.join(models_dir, 'pca.pkl'))
    le          = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    data_info   = joblib.load(os.path.join(models_dir, 'data_info.pkl'))
    checkpoint  = torch.load(os.path.join(models_dir, 'best_model.pth'), map_location='cpu')
    
    model_config = checkpoint.get('model_config', {
        'input_size': 6, 'hidden_size': 64,
        'num_layers': 2, 'num_classes': 2, 'dropout': 0.3
    })
    
    model = FocusLSTM(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes'],
        dropout=model_config.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, scaler, pca, le, data_info, checkpoint

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def run_inference(eeg_df_raw, bpm_df_raw, model, scaler, pca, le, data_info):
    """Run full inference pipeline"""
    
    EEG_COLS     = ['Timestamp', 'Low Alpha', 'High Alpha', 'Low Beta', 'High Beta']
    BPM_COLS     = ['Timestamp', 'BPM', 'Avg BPM']
    FEATURE_COLS = ['Low Alpha', 'High Alpha', 'Low Beta', 'High Beta', 'BPM', 'Avg BPM']
    
    # Filter columns
    eeg_df = eeg_df_raw[[c for c in EEG_COLS if c in eeg_df_raw.columns]].copy()
    bpm_df = bpm_df_raw[[c for c in BPM_COLS if c in bpm_df_raw.columns]].copy()
    
    # Clean timestamps
    eeg_df['ts_key'] = eeg_df['Timestamp'].astype(str).str.replace('"', '').str.replace(',', ' ').str[:19]
    bpm_df['ts_key'] = bpm_df['Timestamp'].astype(str).str[:19]
    
    # Deduplicate BPM
    bpm_clean = bpm_df.drop_duplicates(subset=['ts_key'])
    
    # Merge
    merged = eeg_df.merge(bpm_clean, on='ts_key', how='inner', suffixes=('', '_bpm'))
    
    if merged.empty:
        return None, "No matching timestamps between EEG and BPM!"
    
    # Convert numeric
    for col in FEATURE_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('float32')
    
    data_clean = merged[FEATURE_COLS].dropna()
    
    # Preprocessing
    data_scaled = scaler.transform(data_clean.values)
    data_pca    = pca.transform(data_scaled)
    
    # Windowing
    window_size = data_info.get('window_size', 30)
    windows = [data_pca[i:i+window_size]
               for i in range(0, len(data_pca) - window_size + 1, window_size)]
    
    if not windows:
        return None, f"Insufficient data! Need at least {window_size} samples, got {len(data_pca)}."
    
    X_input = torch.tensor(np.array(windows), dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        outputs       = model(X_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted  = torch.max(outputs, 1)
    
    predictions = le.inverse_transform(predicted.numpy())
    probs       = probabilities.numpy()
    
    # Build results dataframe
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        pred_idx = predicted[i].item()
        results.append({
            'Window': i + 1,
            'Start (s)': round(i * 0.3, 1),
            'End (s)': round((i + 1) * 0.3, 1),
            'Prediction': pred,
            'Confidence (%)': round(prob[pred_idx] * 100, 2),
            'Fokus Prob (%)': round(prob[le.transform(['Fokus'])[0]] * 100 if 'Fokus' in le.classes_ else 0, 2),
            'Tidak Fokus Prob (%)': round(prob[le.transform(['Tidak Fokus'])[0]] * 100 if 'Tidak Fokus' in le.classes_ else 0, 2),
        })
    
    return pd.DataFrame(results), None

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def chart_timeline(df, theme):
    """Per-window prediction timeline"""
    colors = [theme['accent-green'] if p == 'Fokus' else theme['accent-red'] for p in df['Prediction']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Window'],
        y=df['Confidence (%)'],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate='<b>Window %{x}</b><br>%{customdata}<br>Confidence: %{y:.1f}%<extra></extra>',
        customdata=df['Prediction'],
    ))
    fig.update_layout(
        paper_bgcolor=theme['bg-chart'],
        plot_bgcolor=theme['plot-chart'],
        font=dict(family='Space Mono, monospace', color=theme['text-secondary'], size=11),
        title=dict(text='Per-Window Prediction Timeline', font=dict(size=13, color=theme['text-primary'])),
        xaxis=dict(title='Window Number', gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        yaxis=dict(title='Confidence (%)', range=[0, 105], gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        height=300,
    )
    return fig

def chart_probability_line(df, theme):
    """Probability line chart over time"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Window'], y=df['Fokus Prob (%)'],
        name='Fokus', line=dict(color=theme['accent-green'], width=2),
        fill='tozeroy', fillcolor=f"{theme['accent-green']}20",
        hovertemplate='Window %{x}<br>Fokus: %{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['Window'], y=df['Tidak Fokus Prob (%)'],
        name='Tidak Fokus', line=dict(color=theme['accent-red'], width=2),
        fill='tozeroy', fillcolor=f"{theme['accent-red']}20",
        hovertemplate='Window %{x}<br>Tidak Fokus: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dot', line_color=theme['text-secondary'], line_width=1)
    fig.update_layout(
        paper_bgcolor=theme['bg-chart'],
        plot_bgcolor=theme['plot-chart'],
        font=dict(family='Space Mono, monospace', color=theme['text-secondary'], size=11),
        title=dict(text='Probability Over Time', font=dict(size=13, color=theme['text-primary'])),
        xaxis=dict(title='Window Number', gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        yaxis=dict(title='Probability (%)', range=[0, 105], gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor='rgba(0,0,0,0)', bordercolor=theme['border'],
            font=dict(color=theme['text-secondary'], size=10),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
        ),
        height=300,
    )
    return fig

def chart_donut(fokus_pct, tidak_fokus_pct, theme):
    """Donut chart for overall distribution"""
    fig = go.Figure(go.Pie(
        labels=['Fokus', 'Tidak Fokus'],
        values=[fokus_pct, tidak_fokus_pct],
        hole=0.65,
        marker=dict(colors=[theme['accent-green'], theme['accent-red']], line=dict(width=0)),
        textinfo='percent',
        textfont=dict(family='Space Mono', size=12, color=theme['text-primary']),
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    ))
    
    max_pct = max(fokus_pct, tidak_fokus_pct)
    
    fig.update_layout(
        paper_bgcolor=theme['bg-chart'],
        plot_bgcolor=theme['plot-chart'],
        font=dict(family='Space Mono, monospace', color=theme['text-secondary'], size=11),
        title=dict(text='Distribution', font=dict(size=13, color=theme['text-primary'])),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)', 
            font=dict(color=theme['text-secondary'], size=10),
            orientation='h', 
            yanchor='bottom', 
            y=-0.15, 
            xanchor='center', 
            x=0.5
        ),
        height=300,
        annotations=[dict(
            text=f"<b>{max_pct:.0f}%</b>",
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(family='Space Mono', size=22, color=theme['text-primary'])
        )]
    )
    return fig

def chart_confidence_hist(df, theme):
    """Histogram of confidence distribution"""
    fokus_conf  = df[df['Prediction'] == 'Fokus']['Confidence (%)']
    tidak_conf  = df[df['Prediction'] == 'Tidak Fokus']['Confidence (%)']
    
    fig = go.Figure()
    if len(fokus_conf):
        fig.add_trace(go.Histogram(
            x=fokus_conf, name='Fokus',
            marker_color=f"{theme['accent-green']}b3",
            nbinsx=20, hovertemplate='Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))
    if len(tidak_conf):
        fig.add_trace(go.Histogram(
            x=tidak_conf, name='Tidak Fokus',
            marker_color=f"{theme['accent-red']}b3",
            nbinsx=20, hovertemplate='Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))
    fig.update_layout(
        paper_bgcolor=theme['bg-chart'],
        plot_bgcolor=theme['plot-chart'],
        font=dict(family='Space Mono, monospace', color=theme['text-secondary'], size=11),
        title=dict(text='Confidence Distribution', font=dict(size=13, color=theme['text-primary'])),
        xaxis=dict(title='Confidence (%)', gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        yaxis=dict(title='Count', gridcolor=theme['grid-color'], linecolor=theme['text-secondary'], tickfont=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        barmode='overlay',
        legend=dict(
            bgcolor='rgba(0,0,0,0)', font=dict(color=theme['text-secondary'], size=10),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
        ),
        height=300,
    )
    return fig

# ============================================================================
# SIDEBAR - THEME TOGGLE
# ============================================================================

with st.sidebar:
    # Theme toggle
    st.markdown('<div style="margin-bottom: 16px;">', unsafe_allow_html=True)
    col_dark, col_light = st.columns(2, gap="small")
    with col_dark:
        if st.button("🌙 Dark", key="theme_dark", use_container_width=True):
            st.session_state.theme_mode = 'dark'
            st.rerun()
    with col_light:
        if st.button("☀️ Light", key="theme_light", use_container_width=True):
            st.session_state.theme_mode = 'light'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-family: Space Mono, monospace; font-size: 1.3rem; 
                    background: linear-gradient(90deg, {}, {});
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-weight: 700; letter-spacing: -0.5px;">
            🧠 FocusNet
        </div>
        <div style="font-size: 0.72rem; color: {}; margin-top: 4px; 
                    letter-spacing: 1px; text-transform: uppercase;">
            LSTM · EEG + BPM
        </div>
    </div>
    """.format(current_theme['accent-green'], current_theme['accent-blue'], current_theme['text-secondary']), unsafe_allow_html=True)
    
    # Load model status
    try:
        model, scaler, pca, le, data_info, checkpoint = load_artifacts()
        st.markdown("""
        <div style="background:rgba({},{},{},0.08); border:1px solid rgba({},{},{},0.25); 
                    border-radius:8px; padding:12px 16px; font-size:0.8rem;">
            <div style="color:{}; font-family:Space Mono; font-size:0.7rem; 
                        letter-spacing:1px; margin-bottom:8px;">✓ MODEL LOADED</div>
            <div style="color:{};">Epoch: <span style="color:{}">{}</span></div>
            <div style="color:{};">Accuracy: <span style="color:{}">{:.2f}%</span></div>
            <div style="color:{};">Classes: <span style="color:{}">{}</span></div>
        </div>
        """.format(
            *([0,255,136] if st.session_state.theme_mode == 'dark' else [0,168,92]),
            *([0,255,136] if st.session_state.theme_mode == 'dark' else [0,168,92]),
            current_theme['accent-green'],
            current_theme['text-secondary'],
            current_theme['text-primary'],
            checkpoint.get('epoch', 0) + 1,
            current_theme['text-secondary'],
            current_theme['text-primary'],
            checkpoint.get('test_acc', 0),
            current_theme['text-secondary'],
            current_theme['text-primary'],
            ', '.join(le.classes_)
        ), unsafe_allow_html=True)
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Model not found!\n\nPlace model files in `/models/` folder.\n\n`{e}`")
        model_loaded = False
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-family:Space Mono; font-size:0.7rem; letter-spacing:1.5px; 
                color:{}; text-transform:uppercase; margin-bottom:10px;">
        ── UPLOAD DATA ──
    </div>
    """.format(current_theme['text-secondary']), unsafe_allow_html=True)
    
    eeg_file = st.file_uploader("📊 EEG File (.csv)", type=['csv'], key="eeg")
    bpm_file = st.file_uploader("💓 BPM File (.csv)", type=['csv'], key="bpm")
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    run_btn = st.button("▶ RUN ANALYSIS", disabled=(not model_loaded), use_container_width=True)
    
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-family:Space Mono; font-size:0.65rem; color:{}; line-height:1.8;">
        <div style="color:{}; letter-spacing:1px; margin-bottom:8px;">── INFO ──</div>
        <div>Window: 30 timesteps</div>
        <div>Duration/window: 0.3s</div>
        <div>Features: 6 (post-PCA)</div>
        <div>Model: LSTM 2-layer</div>
    </div>
    """.format(current_theme['text-secondary'], current_theme['text-secondary']), unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <p class="header-title">🧠 Identifikasi Tingkat Fokus berbasis Multimodal</p>
    <p class="header-subtitle">EEG + BPM Multimodal Analysis · LSTM Deep Learning</p>
    <p class="header-subtitle">Muhammad Azril Haidar Al Matiin - 23051640011</p>
    <span class="header-badge">Multimodal · TESIS · v1.5</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEFAULT STATE — Instructions
# ============================================================================

if not run_btn or eeg_file is None or bpm_file is None:
    
    if not model_loaded:
        st.warning("⚠️ Model files not found. Please check the `/models/` directory.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card metric-accent-blue">
            <div class="metric-label">Step 1</div>
            <div class="metric-value" style="font-size:1.5rem">📊 Upload</div>
            <div class="metric-sub">Upload EEG CSV dan BPM CSV di sidebar kiri</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card metric-accent-green">
            <div class="metric-label">Step 2</div>
            <div class="metric-value" style="font-size:1.5rem">▶ Run</div>
            <div class="metric-sub">Klik tombol RUN ANALYSIS untuk memulai prediksi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card metric-accent-yellow">
            <div class="metric-label">Step 3</div>
            <div class="metric-value" style="font-size:1.5rem">📈 Hasil</div>
            <div class="metric-sub">Lihat dashboard prediksi, grafik, dan statistik</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-top:24px">
        <b style="color:{accent_blue}">📌 Format File yang Dibutuhkan:</b><br><br>
        <b>EEG CSV:</b> Timestamp, Low Alpha, High Alpha, Low Beta, High Beta<br>
        <b>BPM CSV:</b> Timestamp, BPM, Avg BPM
    </div>
    """.format(accent_blue=current_theme['accent-blue']), unsafe_allow_html=True)
    
    st.stop()

# ============================================================================
# RUN INFERENCE
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
# COMPUTE STATS
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
# VERDICT BANNER
# ============================================================================

if majority == 'Fokus':
    st.markdown(f"""
    <div class="verdict-fokus">
        <div class="verdict-label">Overall Verdict</div>
        <div class="verdict-text-fokus">🟢 FOKUS</div>
        <div style="color:{current_theme['text-secondary']}; font-size:0.85rem; margin-top:8px;">
            Dominant in {fokus_pct:.1f}% of sessions · Avg confidence {avg_conf:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-tidak-fokus">
        <div class="verdict-label">Overall Verdict</div>
        <div class="verdict-text-tidak-fokus">🔴 TIDAK FOKUS</div>
        <div style="color:{current_theme['text-secondary']}; font-size:0.85rem; margin-top:8px;">
            Dominant in {tidak_pct:.1f}% of sessions · Avg confidence {avg_conf:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ============================================================================
# METRIC CARDS
# ============================================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card metric-accent-blue">
        <div class="metric-label">Total Windows</div>
        <div class="metric-value">{total_windows}</div>
        <div class="metric-sub">windows analyzed</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card metric-accent-yellow">
        <div class="metric-label">Total Duration</div>
        <div class="metric-value">{total_duration:.1f}s</div>
        <div class="metric-sub">{total_duration/60:.2f} minutes</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card metric-accent-green">
        <div class="metric-label">Fokus</div>
        <div class="metric-value" style="color:{current_theme['accent-green']}">{fokus_pct:.1f}%</div>
        <div class="metric-sub">{fokus_count} windows · {fokus_count*0.3:.1f}s</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card metric-accent-red">
        <div class="metric-label">Tidak Fokus</div>
        <div class="metric-value" style="color:{current_theme['accent-red']}">{tidak_pct:.1f}%</div>
        <div class="metric-sub">{tidak_count} windows · {tidak_count*0.3:.1f}s</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card metric-accent-blue">
        <div class="metric-label">Avg Confidence</div>
        <div class="metric-value">{avg_conf:.1f}%</div>
        <div class="metric-sub">model certainty</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ============================================================================
# CHARTS - ROW 1
# ============================================================================

st.markdown('<div class="section-header">── VISUALIZATION</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.plotly_chart(chart_timeline(results_df, current_theme), use_container_width=True)

with col_right:
    st.plotly_chart(chart_donut(fokus_pct, tidak_pct, current_theme), use_container_width=True)

# ============================================================================
# TABS: TABLE + DOWNLOAD
# ============================================================================

st.markdown('<div class="section-header">── DETAIL DATA</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 PER-WINDOW RESULTS", "⬇️ EXPORT"])

with tab1:
    def color_prediction(val):
        if val == 'Fokus':
            return f'color: {current_theme["accent-green"]}; font-weight: bold'
        else:
            return f'color: {current_theme["accent-red"]}; font-weight: bold'
    
    # FIX: Use .map() instead of deprecated .applymap() for Pandas 2.x
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
        'font-family': 'Space Mono, monospace',
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
        file_name=f"focus_prediction_results.csv",
        mime="text/csv"
    )
