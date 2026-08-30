"""
Streamlit frontend for Micro-Doppler Radar Threat Detection.
Streams real-time frames directly from the FastAPI WebSocket (/ws/radar).
"""
import streamlit as st
import json
import websocket
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
WS_BACKEND_URL = "ws://localhost:8000/ws/radar"

BG = "#060a0c"
BG_ALT = "#0a1012"
PANEL_BG = "#0d1416"
PANEL_BORDER = "#1e2b2a"
ACCENT = "#00ffb2"
ACCENT_DIM = "#0a5c46"
WARN = "#ff3b3b"
TEXT = "#e6f7f1"
TEXT_DIM = "#7d9d94"

THREAT_STYLE = {
    "drone":   {"color": "#ff3b3b", "icon": "🛸", "label": "DRONE",   "level": "HIGH"},
    "human":   {"color": "#ffb020", "icon": "🚶", "label": "HUMAN",   "level": "MEDIUM"},
    "vehicle": {"color": "#3bb0ff", "icon": "🚗", "label": "VEHICLE", "level": "MEDIUM"},
    "bird":    {"color": "#5fdc8f", "icon": "🐦", "label": "BIRD",    "level": "LOW"},
}

st.set_page_config(page_title="TACTICAL RADAR // MDR-1", layout="wide", page_icon="🎯")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 50% -10%, #0f1c19 0%, {BG} 55%), {BG};
        color: {TEXT};
    }}

    #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}

    .site-nav {{
        position: sticky;
        top: 0;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 56px;
        background: rgba(6,10,12,0.85);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid {PANEL_BORDER};
    }}
    .site-logo {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 18px;
        letter-spacing: 2px;
        color: {TEXT};
    }}
    .site-logo span {{ color: {ACCENT}; }}
    .site-links {{
        display: flex;
        gap: 34px;
        font-size: 13px;
        letter-spacing: 0.5px;
        color: {TEXT_DIM};
    }}
    .site-links a {{ color: {TEXT_DIM}; text-decoration: none; }}
    .site-links .active {{ color: {ACCENT}; }}
    .nav-cta {{
        border: 1px solid {ACCENT_DIM};
        color: {ACCENT};
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 12px;
        letter-spacing: 1.5px;
        background: rgba(0,255,178,0.06);
    }}

    .hero {{
        padding: 40px 56px 20px 56px;
        text-align: center;
        position: relative;
    }}
    .hero-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid {ACCENT_DIM};
        color: {ACCENT};
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        letter-spacing: 2px;
        background: rgba(0,255,178,0.06);
        margin-bottom: 14px;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: {ACCENT};
        box-shadow: 0 0 8px {ACCENT};
    }}
    .hero-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 38px;
        letter-spacing: 1px;
        margin: 0 auto 10px auto;
        background: linear-gradient(180deg, #ffffff 10%, {ACCENT} 120%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        font-size: 14px;
        color: {TEXT_DIM};
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.5;
    }}

    .section-wrap {{ padding: 10px 56px 40px 56px; }}
    .site-card {{
        border: 1px solid {PANEL_BORDER};
        background: linear-gradient(180deg, {PANEL_BG} 0%, {BG_ALT} 100%);
        border-radius: 14px;
        padding: 18px 20px 6px 20px;
        margin-bottom: 20px;
    }}
    .card-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Orbitron', sans-serif;
        font-size: 13px;
        letter-spacing: 1.5px;
        color: {TEXT};
        margin-bottom: 4px;
    }}
    .card-title .dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        background: {ACCENT};
        box-shadow: 0 0 6px {ACCENT};
    }}
    .card-desc {{
        font-size: 12px;
        color: {TEXT_DIM};
        margin-bottom: 10px;
        border-bottom: 1px solid {PANEL_BORDER};
        padding-bottom: 10px;
    }}

    .threat-card {{
        border: 1px solid var(--tc-color, {ACCENT});
        border-left: 6px solid var(--tc-color, {ACCENT});
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 16px 18px;
    }}
    .threat-name {{
        font-family: 'Orbitron', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--tc-color, {ACCENT});
        letter-spacing: 2px;
        margin: 0 0 8px 0;
    }}
    .threat-row {{
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        padding: 5px 0;
        border-bottom: 1px dashed {PANEL_BORDER};
        color: {TEXT};
    }}
    .threat-row span.label {{ color: {TEXT_DIM}; letter-spacing: 1px; }}
    .threat-level-badge {{
        display: inline-block;
        margin-top: 12px;
        font-size: 11px;
        letter-spacing: 2px;
        padding: 4px 12px;
        border-radius: 12px;
        border: 1px solid var(--tc-color, {ACCENT});
        color: var(--tc-color, {ACCENT});
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Static Headers
# -----------------------------------------------------------------------------
now_str = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"""
    <div class="site-nav">
        <div class="site-logo">◈ MDR<span>-1</span> RADAR SYSTEMS</div>
        <div class="site-links">
            <a href="#" class="active">Overview</a>
            <a href="#">Sweep</a>
            <a href="#">Spectrogram</a>
            <a href="#">Classification</a>
        </div>
        <div class="nav-cta">● WS STREAM ACTIVE</div>
    </div>
    <div class="hero">
        <div class="hero-eyebrow"><span class="status-dot"></span>LIVE WEBSOCKET STREAM</div>
        <div class="hero-title">Micro-Doppler Radar<br/>Threat Detection Platform</div>
        <div class="hero-subtitle">
            Real-time range-Doppler sweeps, spectrogram analysis, and automated
            threat classification streamed directly from FastAPI backend.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Share Tech Mono"),
    margin=dict(l=10, r=10, t=10, b=10),
)

dashboard_placeholder = st.empty()

# -----------------------------------------------------------------------------
# WebSocket Stream Consumer
# -----------------------------------------------------------------------------
try:
    ws = websocket.create_connection(WS_BACKEND_URL)
    while True:
        raw_msg = ws.recv()
        frame = json.loads(raw_msg)

        with dashboard_placeholder.container():
            st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])

            # 1. Range / Velocity Sweep
            with col1:
                st.markdown('<div class="site-card">', unsafe_allow_html=True)
                st.markdown(
                    '<p class="card-title"><span class="dot"></span>RANGE / VELOCITY SWEEP</p>'
                    '<p class="card-desc">Downsampled range-Doppler map from live sensor frame.</p>',
                    unsafe_allow_html=True,
                )

                range_doppler = np.array(frame["range_doppler"])
                rows, cols = range_doppler.shape
                points = []
                for i in range(0, rows, 2):
                    for j in range(0, cols, 2):
                        range_km = (i / rows) * 5.0
                        velocity = (j / cols) * 60.0 - 20.0
                        intensity = range_doppler[i, j]
                        points.append({"range": range_km, "velocity": velocity, "intensity": intensity})

                df_rd = pd.DataFrame(points)
                fig_rd = px.scatter(
                    df_rd,
                    x="range",
                    y="velocity",
                    color="intensity",
                    color_continuous_scale=[[0, "#04231c"], [0.5, ACCENT_DIM], [1, ACCENT]],
                    range_x=[0, 5],
                    range_y=[-20, 40],
                    height=280,
                )
                fig_rd.update_traces(marker=dict(size=6, line=dict(width=0)))
                fig_rd.update_layout(**PLOTLY_LAYOUT)
                fig_rd.update_xaxes(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER)
                fig_rd.update_yaxes(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER)
                st.plotly_chart(fig_rd, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            # 2. Spectrogram Heatmap
            with col2:
                st.markdown('<div class="site-card">', unsafe_allow_html=True)
                st.markdown(
                    '<p class="card-title"><span class="dot"></span>DOPPLER SPECTROGRAM</p>'
                    '<p class="card-desc">Time vs frequency energy distribution.</p>',
                    unsafe_allow_html=True,
                )

                spectrogram = np.array(frame["spectrogram"])
                fig_spec = px.imshow(
                    spectrogram,
                    color_continuous_scale=[[0, "#04120f"], [0.5, "#0a5c46"], [1, "#00ffb2"]],
                    aspect="auto",
                    height=280,
                )
                fig_spec.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_spec, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. Threat Alert Card
            with col3:
                st.markdown('<div class="site-card">', unsafe_allow_html=True)
                st.markdown(
                    '<p class="card-title"><span class="dot"></span>THREAT CLASSIFICATION</p>'
                    '<p class="card-desc">Automated classification of return target.</p>',
                    unsafe_allow_html=True,
                )

                threat = frame["threat"]
                style = THREAT_STYLE.get(threat["class"], THREAT_STYLE["bird"])

                st.markdown(
                    f"""
                    <div class="threat-card" style="--tc-color:{style['color']};">
                        <p class="threat-name">{style['icon']} {style['label']}</p>
                        <div class="threat-row"><span class="label">CONFIDENCE</span><span>{threat['confidence']*100:.1f}%</span></div>
                        <div class="threat-row"><span class="label">BEARING</span><span>{threat['bearing']:.1f}°</span></div>
                        <div class="threat-row"><span class="label">RANGE</span><span>{threat['range_km']:.2f} km</span></div>
                        <div class="threat-row"><span class="label">VELOCITY</span><span>{threat['velocity_mps']:.1f} m/s</span></div>
                        <span class="threat-level-badge" style="--tc-color:{style['color']};">THREAT LEVEL: {style['level']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"WebSocket Link Error: Unable to connect to {WS_BACKEND_URL}. Ensure FastAPI is running on port 8000. ({e})")