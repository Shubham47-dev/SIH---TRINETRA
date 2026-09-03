import streamlit as st
import requests
import time
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TRINETRA | Command Center", 
    layout="wide", 
    page_icon="👁️"
)

# Tactical CSS styling for Threat Levels
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .threat-red { color: #FF4B4B; font-size: 38px; font-weight: 900; text-align: center; background-color: rgba(255, 75, 75, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #FF4B4B;}
    .threat-orange { color: #FFA500; font-size: 38px; font-weight: 900; text-align: center; background-color: rgba(255, 165, 0, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #FFA500;}
    .threat-yellow { color: #FFD700; font-size: 38px; font-weight: bold; text-align: center; background-color: rgba(255, 215, 0, 0.1); padding: 20px; border-radius: 10px;}
    .threat-green { color: #00CC96; font-size: 38px; font-weight: bold; text-align: center; padding: 20px; }
    .telemetry-text { text-align: center; color: #888888; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ TRINETRA: Acoustic Micro-Doppler C-UAS")
st.markdown("### 📡 Live Sensor Telemetry (Edge Node 01)")
st.markdown("---")

# --- 2. DYNAMIC PLACEHOLDERS ---
# These empty containers get overwritten every second to prevent the page from flickering
status_placeholder = st.empty()
chart_placeholder = st.empty()

def fetch_telemetry():
    """Pulls the live state from the FastAPI Hub."""
    try:
        response = requests.get("http://127.0.0.1:8000/latest-threat", timeout=1)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

# --- 3. RENDERING LOOP ---
data = fetch_telemetry()

if data:
    target = data.get("target_identified", "Unknown")
    confidence = data.get("confidence", 0.0)
    timestamp = data.get("timestamp", "N/A")
    details = data.get("details", {})

    # TACTICAL COLOR LOGIC
    if target == "Drone (Payload)":
        css_class = "threat-red"
        alert_msg = f"🚨 CRITICAL THREAT: {target.upper()} [{confidence}%]"
        chart_color = "#FF4B4B"
    elif target == "Drone (Unarmed)":
        css_class = "threat-orange"
        alert_msg = f"⚠️ INTRUSION DETECTED: {target.upper()} [{confidence}%]"
        chart_color = "#FFA500"
    elif target == "Acoustic Anomaly (Unverified)":
        css_class = "threat-yellow"
        alert_msg = f"🔍 ANOMALY: {target.upper()} [{confidence}%]"
        chart_color = "#FFD700"
    else:
        css_class = "threat-green"
        alert_msg = f"✅ CLEAR: {target.upper()} [{confidence}%]"
        chart_color = "#00CC96"

    # UPDATE THE BIG BANNER
    with status_placeholder.container():
        st.markdown(f"<div class='{css_class}'>{alert_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='telemetry-text'>Last Packet Received: {timestamp}</p>", unsafe_allow_html=True)

    # UPDATE THE 5-CLASS BAR CHART
    with chart_placeholder.container():
        if details:
            st.markdown("#### Acoustic Signature Probabilities")
            # Convert dictionary to Pandas DataFrame for Streamlit's native charting
            df = pd.DataFrame({
                "Classification": list(details.keys()),
                "Probability (%)": list(details.values())
            }).set_index("Classification")
            
            st.bar_chart(df, color=chart_color, height=350)
else:
    with status_placeholder.container():
        st.error("🔌 OFFLINE: Cannot connect to TRINETRA Command Hub. Ensure `hub_server.py` is running on port 8000.")

# --- 4. ENGINE CYCLE ---
# Pauses for exactly 1 second, then re-runs the entire script to pull new data
time.sleep(1.0)
st.rerun()