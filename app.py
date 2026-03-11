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
# THEME SYSTEM - Initialize Session State
# ============================================================================

# Initialize theme in session state if not exists
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default theme

# ============================================================================
# THEME DEFINITIONS
# ============================================================================

THEMES = {
    'dark': {
        'bg_primary': '#0a0e1a',
        'bg_secondary': '#111827',
        'bg_card': '#1a2035',
        'accent_green': '#00ff88',
        'accent_red': '#ff4757',
        'accent_blue': '#4facfe',
        'accent_yellow': '#ffd32a',
        'text_primary': '#e8eaf6',
        'text_secondary': '#8892b0',
        'border': '#2d3561',
        'shadow': 'rgba(0,0,0,0.3)',
        'gradient_start': '#0a0e1a',
        'gradient_mid': '#1a2035',
        'gradient_end': '#0d1b2a',
    },
    'light': {
        'bg_primary': '#f8f9fa',
        'bg_secondary': '#ffffff',
        'bg_card': '#ffffff',
        'accent_green': '#00c853',
        'accent_red': '#d32f2f',
        'accent_blue': '#1976d2',
        'accent_yellow': '#ffa000',
        'text_primary': '#212121',
        'text_secondary': '#616161',
        'border': '#e0e0e0',
        'shadow': 'rgba(0,0,0,0.1)',
        'gradient_start': '#f5f5f5',
        'gradient_mid': '#ffffff',
        'gradient_end': '#fafafa',
    }
}

# Get current theme
theme = THEMES[st.session_state.theme]

# ============================================================================
# DYNAMIC CSS BASED ON THEME
# ============================================================================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    
    :root {{
        --bg-primary: {theme['bg_primary']};
        --bg-secondary: {theme['bg_secondary']};
        --bg-card: {theme['bg_card']};
        --accent-green: {theme['accent_green']};
        --accent-red: {theme['accent_red']};
        --accent-blue: {theme['accent_blue']};
        --accent-yellow: {theme['accent_yellow']};
        --text-primary: {theme['text_primary']};
        --text-secondary: {theme['text_secondary']};
        --border: {theme['border']};
        --shadow: {theme['shadow']};
    }}
    
    /* Main background */
    .stApp {{
        background: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }}
    
    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }}
    
    [data-testid="stSidebar"] * {{
        color: var(--text-primary) !important;
    }}
    
    /* Theme Toggle Button */
    .theme-toggle {{
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 999999;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 8px 16px;
        cursor: pointer;
        box-shadow: 0 4px 12px var(--shadow);
        transition: all 0.3s ease;
    }}
    
    .theme-toggle:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--shadow);
    }}
    
    .theme-toggle-icon {{
        font-size: 20px;
        margin-right: 8px;
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {theme['gradient_start']} 0%, {theme['gradient_mid']} 50%, {theme['gradient_end']} 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 12px var(--shadow);
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, {'rgba(0,255,136,0.06)' if st.session_state.theme == 'dark' else 'rgba(25,118,210,0.08)'} 0%, transparent 70%);
        border-radius: 50%;
    }}
    
    .main-header::after {{
        content: '';
        position: absolute;
        bottom: -50%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, {'rgba(79,172,254,0.06)' if st.session_state.theme == 'dark' else 'rgba(0,200,83,0.08)'} 0%, transparent 70%);
        border-radius: 50%;
    }}
    
    .header-title {{
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }}
    
    .header-subtitle {{
        color: var(--text-secondary);
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }}
    
    /* File Upload Section */
    .upload-section {{
        background: var(--bg-card);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }}
    
    .upload-section:hover {{
        border-color: var(--accent-blue);
        background: {'#1e2744' if st.session_state.theme == 'dark' else '#f5f5f5'};
    }}
    
    /* Verdict Banner */
    .verdict-banner {{
        border-radius: 16px;
        padding: 24px 32px;
        margin: 24px 0;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 24px var(--shadow);
        animation: slideIn 0.5s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .verdict-fokus {{
        background: linear-gradient(135deg, {'#00ff8840' if st.session_state.theme == 'dark' else '#e8f5e9'} 0%, {'#00ff8820' if st.session_state.theme == 'dark' else '#c8e6c9'} 100%);
        border: 2px solid var(--accent-green);
        color: var(--accent-green);
    }}
    
    .verdict-tidak-fokus {{
        background: linear-gradient(135deg, {'#ff475740' if st.session_state.theme == 'dark' else '#ffebee'} 0%, {'#ff475720' if st.session_state.theme == 'dark' else '#ffcdd2'} 100%);
        border: 2px solid var(--accent-red);
        color: var(--accent-red);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px var(--shadow);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 16px var(--shadow);
        border-color: var(--accent-blue);
    }}
    
    .metric-label {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 4px;
    }}
    
    .metric-unit {{
        font-size: 0.9rem;
        color: var(--text-secondary);
    }}
    
    /* Charts */
    .chart-container {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px var(--shadow);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px var(--shadow);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--shadow);
    }}
    
    /* Data Table */
    .dataframe {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}
    
    .dataframe th {{
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--border) !important;
    }}
    
    .dataframe td {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border) !important;
    }}
    
    /* Input fields */
    .stTextInput input, .stSelectbox select {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: var(--bg-card);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 16px;
    }}
    
    [data-testid="stFileUploader"] label {{
        color: var(--text-primary) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# THEME TOGGLE FUNCTION
# ============================================================================

def toggle_theme():
    """Toggle between dark and light theme"""
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.rerun()

# ============================================================================
# HEADER WITH THEME TOGGLE
# ============================================================================

# Theme toggle button in sidebar
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col2:
        theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        if st.button(theme_icon, help="Toggle theme", use_container_width=True):
            toggle_theme()

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="header-title">🧠 Identifikasi Tingkat Fokus berbasis Multimodal</h1>
    <p class="header-subtitle">Deep Learning-based EEG + BPM Analysis for Student Focus Classification</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class FocusLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
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
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_and_artifacts():
    """Load model, scaler, PCA, and other artifacts"""
    try:
        # Get paths
        models_dir = os.path.join(os.getcwd(), 'models')
        
        # Load data_info
        data_info_path = os.path.join(models_dir, 'data_info.pkl')
        data_info = joblib.load(data_info_path)
        
        # Load preprocessing artifacts
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        pca = joblib.load(os.path.join(models_dir, 'pca.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        
        # Load model
        checkpoint = torch.load(
            os.path.join(models_dir, 'best_model.pth'),
            map_location=torch.device('cpu')
        )
        
        # Get model config
        model_config = checkpoint.get('model_config', {
            'input_size': 6,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': 2,
            'dropout': 0.3
        })
        
        # Initialize model
        model = FocusLSTM(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return {
            'model': model,
            'scaler': scaler,
            'pca': pca,
            'label_encoder': label_encoder,
            'data_info': data_info
        }
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_data(eeg_df, bpm_df, scaler, pca, window_size=30):
    """Preprocess EEG and BPM data for inference"""
    try:
        # Merge data
        eeg_df['ts_key'] = eeg_df['Timestamp'].astype(str).str[:19]
        bpm_df['ts_key'] = bpm_df['Timestamp'].astype(str).str[:19]
        
        # Remove duplicates in BPM
        bpm_df = bpm_df.drop_duplicates(subset='ts_key', keep='first')
        
        # Merge
        merged = pd.merge(eeg_df, bpm_df, on='ts_key', how='inner', suffixes=('_eeg', '_bpm'))
        
        if len(merged) == 0:
            st.error("No matching timestamps between EEG and BPM data!")
            return None
        
        # Select features
        feature_cols = ['Low Alpha', 'High Alpha', 'Low Beta', 'High Beta', 'BPM', 'Avg BPM']
        features = merged[feature_cols].values
        
        # Normalize
        features_scaled = scaler.transform(features)
        
        # PCA
        features_pca = pca.transform(features_scaled)
        
        # Create windows
        num_samples = len(features_pca)
        num_windows = num_samples - window_size + 1
        
        if num_windows <= 0:
            st.error(f"Not enough data for windowing. Need at least {window_size} samples.")
            return None
        
        windows = []
        for i in range(num_windows):
            window = features_pca[i:i+window_size]
            windows.append(window)
        
        windows = np.array(windows)
        
        return windows, merged
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def predict(model, windows, label_encoder):
    """Run prediction on windows"""
    try:
        with torch.no_grad():
            X_tensor = torch.FloatTensor(windows)
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()
            predictions = torch.argmax(outputs, dim=1).numpy()
            labels = label_encoder.inverse_transform(predictions)
            
        return labels, probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# ============================================================================
# PLOTLY THEME CONFIGURATION
# ============================================================================

def get_plotly_theme():
    """Get plotly theme based on current theme"""
    if st.session_state.theme == 'dark':
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': '#1a2035',
            'plot_bgcolor': '#1a2035',
            'font_color': '#e8eaf6',
            'gridcolor': '#2d3561'
        }
    else:
        return {
            'template': 'plotly_white',
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#ffffff',
            'font_color': '#212121',
            'gridcolor': '#e0e0e0'
        }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_timeline_chart(predictions, probabilities):
    """Create per-window timeline chart"""
    plot_theme = get_plotly_theme()
    
    colors = [theme['accent_green'] if p == 'Fokus' else theme['accent_red'] for p in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(predictions))),
        y=[1] * len(predictions),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='<b>Window %{x}</b><br>Prediction: %{customdata[0]}<br>Confidence: %{customdata[1]:.1f}%<extra></extra>',
        customdata=[[predictions[i], probabilities[i].max() * 100] for i in range(len(predictions))]
    ))
    
    fig.update_layout(
        title=dict(text='Per-Window Predictions Timeline', font=dict(size=16, color=plot_theme['font_color'])),
        xaxis=dict(title='Window Number', gridcolor=plot_theme['gridcolor'], color=plot_theme['font_color']),
        yaxis=dict(title='', showticklabels=False, gridcolor=plot_theme['gridcolor']),
        template=plot_theme['template'],
        paper_bgcolor=plot_theme['paper_bgcolor'],
        plot_bgcolor=plot_theme['plot_bgcolor'],
        font=dict(color=plot_theme['font_color']),
        hovermode='x unified',
        height=300,
        showlegend=False,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return fig

def create_distribution_chart(predictions):
    """Create prediction distribution donut chart"""
    plot_theme = get_plotly_theme()
    
    counts = pd.Series(predictions).value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.5,
        marker=dict(colors=[theme['accent_green'], theme['accent_red']]),
        textinfo='label+percent',
        textfont=dict(size=14, color=plot_theme['font_color']),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text='Prediction Distribution', font=dict(size=16, color=plot_theme['font_color'])),
        template=plot_theme['template'],
        paper_bgcolor=plot_theme['paper_bgcolor'],
        font=dict(color=plot_theme['font_color']),
        height=400,
        showlegend=True,
        legend=dict(font=dict(color=plot_theme['font_color'])),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 📁 Upload Data Files")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    eeg_file = st.file_uploader("📊 EEG Data (CSV)", type=['csv'], key='eeg')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    bpm_file = st.file_uploader("❤️ BPM Data (CSV)", type=['csv'], key='bpm')
    st.markdown('</div>', unsafe_allow_html=True)
    
    analyze_button = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.markdown(f"""
    <div style='background: var(--bg-card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);'>
        <p style='margin: 0; color: var(--text-secondary); font-size: 0.9rem;'>
            <b>Architecture:</b> LSTM<br>
            <b>Input:</b> 6 PCA features<br>
            <b>Window Size:</b> 30 timesteps<br>
            <b>Classes:</b> Fokus, Tidak Fokus<br>
            <b>Theme:</b> {st.session_state.theme.capitalize()} Mode
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Load model
artifacts = load_model_and_artifacts()

if artifacts is None:
    st.error("Failed to load model artifacts. Please check the models directory.")
    st.stop()

# Run analysis
if analyze_button and eeg_file and bpm_file:
    with st.spinner("🔄 Processing data..."):
        # Load data
        eeg_df = pd.read_csv(eeg_file)
        bpm_df = pd.read_csv(bpm_file)
        
        # Preprocess
        window_size = artifacts['data_info'].get('window_size', 30)
        result = preprocess_data(
            eeg_df, bpm_df,
            artifacts['scaler'],
            artifacts['pca'],
            window_size
        )
        
        if result is not None:
            windows, merged_data = result
            
            # Predict
            predictions, probabilities = predict(
                artifacts['model'],
                windows,
                artifacts['label_encoder']
            )
            
            if predictions is not None:
                # Calculate statistics
                total_windows = len(predictions)
                fokus_count = np.sum(predictions == 'Fokus')
                tidak_fokus_count = np.sum(predictions == 'Tidak Fokus')
                fokus_pct = (fokus_count / total_windows) * 100
                tidak_fokus_pct = (tidak_fokus_count / total_windows) * 100
                avg_confidence = np.mean(probabilities.max(axis=1)) * 100
                
                # Duration (assuming 10ms per sample)
                duration_ms = len(merged_data) * 10
                duration_sec = duration_ms / 1000
                
                # Overall verdict
                overall_verdict = "Fokus" if fokus_pct > 50 else "Tidak Fokus"
                verdict_class = "verdict-fokus" if overall_verdict == "Fokus" else "verdict-tidak-fokus"
                verdict_emoji = "🟢" if overall_verdict == "Fokus" else "🔴"
                
                # Display verdict banner
                st.markdown(f"""
                <div class="verdict-banner {verdict_class}">
                    {verdict_emoji} <b>OVERALL VERDICT: {overall_verdict.upper()}</b>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Windows</div>
                        <div class="metric-value">{total_windows:,}</div>
                        <div class="metric-unit">windows</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Duration</div>
                        <div class="metric-value">{duration_sec:.1f}</div>
                        <div class="metric-unit">seconds</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {theme['accent_green']};">
                        <div class="metric-label">🟢 Fokus</div>
                        <div class="metric-value" style="color: {theme['accent_green']};">{fokus_pct:.1f}</div>
                        <div class="metric-unit">% of windows</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {theme['accent_red']};">
                        <div class="metric-label">🔴 Tidak Fokus</div>
                        <div class="metric-value" style="color: {theme['accent_red']};">{tidak_fokus_pct:.1f}</div>
                        <div class="metric-unit">% of windows</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Confidence</div>
                        <div class="metric-value">{avg_confidence:.1f}</div>
                        <div class="metric-unit">%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                st.markdown("### 📈 Visualizations")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_timeline = create_timeline_chart(predictions, probabilities)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_dist = create_distribution_chart(predictions)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                
                results_df = pd.DataFrame({
                    'Window': range(len(predictions)),
                    'Prediction': predictions,
                    'Confidence': [f"{p.max()*100:.2f}%" for p in probabilities],
                    'Fokus Prob': [f"{p[artifacts['label_encoder'].transform(['Fokus'])[0]]*100:.2f}%" for p in probabilities],
                    'Tidak Fokus Prob': [f"{p[artifacts['label_encoder'].transform(['Tidak Fokus'])[0]]*100:.2f}%" for p in probabilities]
                })
                
                # Color coding function
                def highlight_predictions(row):
                    if row['Prediction'] == 'Fokus':
                        return [f'background-color: {theme["accent_green"]}20; color: {theme["text_primary"]}'] * len(row)
                    else:
                        return [f'background-color: {theme["accent_red"]}20; color: {theme["text_primary"]}'] * len(row)
                
                styled_df = results_df.style.apply(highlight_predictions, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="focus_detection_results.csv",
                    mime="text/csv"
                )

elif analyze_button:
    st.warning("⚠️ Please upload both EEG and BPM data files first!")

else:
    st.info("👈 Upload EEG and BPM files in the sidebar and click 'RUN ANALYSIS' to begin.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: var(--text-secondary); padding: 20px;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        🧠 Focus Detection System | Deep Learning-based Analysis<br>
        Current Theme: <b>{st.session_state.theme.capitalize()}</b>
    </p>
</div>
""", unsafe_allow_html=True)
