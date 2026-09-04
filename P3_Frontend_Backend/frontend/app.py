"""
Streamlit frontend for Micro-Doppler Radar Threat Detection.
Polls the backend REST endpoint (/latest_frame) every 0.5 seconds and displays:
- Spectrogram heatmap
- Threat classification alerts

Styled as a full defense-tech product website: top nav bar, hero section,
card-based feature grid, and a real site footer — rather than a raw dashboard.
"""
import streamlit as st
import requests
import numpy as np
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
REFRESH_INTERVAL_MS = 500  

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

st_autorefresh(interval=REFRESH_INTERVAL_MS, key="radar_refresh")

# Global styling
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 50% -10%, #0f1c19 0%, {BG} 55%),
            {BG};
        color: {TEXT};
    }}

    /* Kill Streamlit chrome so it reads like a real site, not an app shell */
    #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}

    /* ---------------- NAV BAR ---------------- */
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

    /* ---------------- HERO ---------------- */
    .hero {{
        padding: 64px 56px 40px 56px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: "";
        position: absolute;
        top: -100px; left: 50%;
        transform: translateX(-50%);
        width: 700px; height: 400px;
        background: radial-gradient(circle, rgba(0,255,178,0.14) 0%, transparent 70%);
        pointer-events: none;
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
        margin-bottom: 22px;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: {ACCENT};
        box-shadow: 0 0 8px {ACCENT};
        animation: pulse 1.4s infinite;
    }}
    @keyframes pulse {{
        0%   {{ opacity: 1; }}
        50%  {{ opacity: 0.25; }}
        100% {{ opacity: 1; }}
    }}
    .hero-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 44px;
        letter-spacing: 1px;
        line-height: 1.15;
        margin: 0 auto 16px auto;
        max-width: 900px;
        background: linear-gradient(180deg, #ffffff 10%, {ACCENT} 120%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        font-size: 15px;
        color: {TEXT_DIM};
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    .hero-stats {{
        display: flex;
        justify-content: center;
        gap: 48px;
        margin-top: 40px;
    }}
    .hero-stat-num {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 26px;
        color: {ACCENT};
    }}
    .hero-stat-label {{
        font-size: 11px;
        letter-spacing: 1.5px;
        color: {TEXT_DIM};
        margin-top: 2px;
    }}

    /* ---------------- SECTION LABEL ---------------- */
    .section-wrap {{ padding: 10px 56px 60px 56px; }}
    .section-header {{
        text-align: center;
        margin-bottom: 30px;
    }}
    .section-kicker {{
        font-size: 11px;
        letter-spacing: 3px;
        color: {ACCENT};
        text-transform: uppercase;
    }}
    .section-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        color: {TEXT};
        margin-top: 8px;
    }}

    /* ---------------- FEATURE CARDS ---------------- */
    .site-card {{
        border: 1px solid {PANEL_BORDER};
        background: linear-gradient(180deg, {PANEL_BG} 0%, {BG_ALT} 100%);
        border-radius: 14px;
        padding: 20px 22px 6px 22px;
        margin-bottom: 24px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }}
    .site-card:hover {{
        border-color: {ACCENT_DIM};
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
        margin-bottom: 12px;
        border-bottom: 1px solid {PANEL_BORDER};
        padding-bottom: 12px;
    }}

    /* Threat card */
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

    /* ---------------- SITE FOOTER ---------------- */
    .site-footer {{
        border-top: 1px solid {PANEL_BORDER};
        padding: 40px 56px 30px 56px;
        display: flex;
        justify-content: space-between;
        gap: 40px;
        background: {BG_ALT};
    }}
    .footer-col h4 {{
        font-family: 'Orbitron', sans-serif;
        font-size: 12px;
        letter-spacing: 1.5px;
        color: {TEXT};
        margin-bottom: 12px;
    }}
    .footer-col p, .footer-col div {{
        font-size: 12px;
        color: {TEXT_DIM};
        line-height: 2;
    }}
    .footer-bottom {{
        text-align: center;
        font-size: 11px;
        letter-spacing: 1px;
        color: {TEXT_DIM};
        padding: 18px 0 8px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Nav bar
now_str = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"""
    <div class="site-nav">
        <div class="site-logo">◈ MDR<span>-1</span> RADAR SYSTEMS</div>
        <div class="site-links">
            <a href="#" class="active">Overview</a>
            <a href="#">Spectrogram</a>
            <a href="#">Classification</a>
            <a href="#">Docs</a>
        </div>
        <div class="nav-cta">● LIVE &nbsp;{now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero section
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow"><span class="status-dot"></span>SECTOR SWEEP ACTIVE</div>
        <div class="hero-title">Micro-Doppler Radar<br/>Threat Detection Platform</div>
        <div class="hero-subtitle">
            Real-time spectrogram analysis and automated threat classification
            streamed live from the sensor backend.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=0.5)
def get_latest_frame():
    try:
        response = requests.get(f"{BACKEND_URL}/latest_frame", timeout=1)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

frame = get_latest_frame()

if frame is None:
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="site-card" style="border-color:{WARN};">
                <p class="card-title" style="color:{WARN};">⚠ LINK DOWN</p>
                <p style="color:{TEXT}; padding-bottom: 16px;">
                    Cannot reach backend at <code>{BACKEND_URL}</code>. Confirm the FastAPI
                    service is running on port 8000 and retry.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Share Tech Mono"),
    margin=dict(l=10, r=10, t=10, b=10),
)

st.markdown(
    """
    <div class="section-wrap" style="padding-bottom: 10px;">
        <div class="section-header">
            <div class="section-kicker">Live sensor output</div>
            <div class="section-title">Spectrogram · Classification</div>
        </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="site-card">', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-title"><span class="dot"></span>DOPPLER SPECTROGRAM</p>'
        '<p class="card-desc">Time vs frequency energy distribution for the current frame.</p>',
        unsafe_allow_html=True,
    )

    spectrogram = np.array(frame["spectrogram"])
    fig_spec = px.imshow(
        spectrogram,
        color_continuous_scale=[[0, "#04120f"], [0.5, "#0a5c46"], [1, "#00ffb2"]],
        aspect="auto",
        labels=dict(x="Time bin", y="Frequency bin", color="Intensity"),
        height=300,
    )
    fig_spec.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_spec, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="site-card">', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-title"><span class="dot"></span>THREAT CLASSIFICATION</p>'
        '<p class="card-desc">Automated classification of the strongest return in view.</p>',
        unsafe_allow_html=True,
    )

    threat = frame["threat"]
    threat_class = threat["class"]
    confidence = threat["confidence"]
    bearing = threat["bearing"]
    range_km = threat["range_km"]
    velocity = threat["velocity_mps"]

    style = THREAT_STYLE.get(threat_class, THREAT_STYLE["bird"])

    st.markdown(
        f"""
        <div class="threat-card" style="--tc-color:{style['color']};">
            <p class="threat-name">{style['icon']} {style['label']}</p>
            <div class="threat-row"><span class="label">CONFIDENCE</span><span>{confidence*100:.1f}%</span></div>
            <div class="threat-row"><span class="label">BEARING</span><span>{bearing:.1f}°</span></div>
            <div class="threat-row"><span class="label">RANGE</span><span>{range_km:.2f} km</span></div>
            <div class="threat-row"><span class="label">VELOCITY</span><span>{velocity:.1f} m/s</span></div>
            <span class="threat-level-badge" style="--tc-color:{style['color']};">THREAT LEVEL: {style['level']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .section-wrap

# Site footer
st.markdown(
    f"""
    <div class="site-footer">
        <div class="footer-col" style="flex: 1.4;">
            <h4>◈ MDR-1 RADAR SYSTEMS</h4>
            <p>Micro-Doppler sensor console for spectrogram analysis and live
            threat classification.</p>
        </div>
        <div class="footer-col">
            <h4>SYSTEM</h4>
            <div>Spectrogram</div>
            <div>Classification</div>
        </div>
        <div class="footer-col">
            <h4>STATUS</h4>
            <div>Backend: {BACKEND_URL}</div>
            <div>Refresh: {REFRESH_INTERVAL_MS} ms</div>
            <div>Last frame: {frame['timestamp']}</div>
        </div>
        <div class="footer-col">
            <h4>BUILD</h4>
            <div>Console v1.2</div>
            <div>Streamlit frontend</div>
        </div>
    </div>
    <div class="footer-bottom">MDR-1 CONSOLE — INTERNAL DEMO ENVIRONMENT</div>
    """,
    unsafe_allow_html=True,
)
