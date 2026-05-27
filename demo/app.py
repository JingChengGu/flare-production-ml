"""
FLARE Demo App v2 — Light theme, high contrast
Run: streamlit run demo/app.py
"""

import streamlit as st
import requests
from PIL import Image

API_URL = "https://flare-api-610805014879.us-central1.run.app"

COMPONENT_LABELS = {
    "screen": "Screen",
    "body":   "Body / OOS",
    "cable":  "Cable",
    "plug":   "Plug",
}

st.set_page_config(
    page_title="FLARE — EV Fault Detection",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #f5f5f0; }
    #MainMenu, footer, header { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    .stButton > button {
        background: #1a1a1a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover { background: #333333 !important; }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 8px;
    }

    [data-testid="stNotificationContentWarning"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
        <div style="font-size:22px; font-weight:800; color:#1a1a1a; letter-spacing:-0.5px;">⚡ FLARE</div>
        <div style="font-size:12px; color:#888; margin-top:4px;">EV Fault Detection System</div>
    </div>
    <hr style="border:none; border-top:1px solid #eeeeee; margin:12px 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;">System Info</div>
    """, unsafe_allow_html=True)

    info_items = [
        ("Models", "SegFormer B3 + ViT × 4"),
        ("Runtime", "ONNX Runtime (CPU)"),
        ("Deployment", "GCP Cloud Run"),
        ("Region", "us-central1"),
    ]
    for label, value in info_items:
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="font-size:11px; color:#aaa; margin-bottom:2px;">{label}</div>
            <div style="font-size:13px; color:#1a1a1a; font-weight:500;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none; border-top:1px solid #eeeeee; margin:12px 0;">
    <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:10px 12px;">
        <div style="font-size:11px; font-weight:700; color:#92400e; margin-bottom:3px;">Cold Start Note</div>
        <div style="font-size:11px; color:#78350f;">First request after idle may take 60–90s. Subsequent requests ~5s.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    charger_id = st.text_input("Charger ID (optional)", placeholder="e.g. CH-1234")

# ---------------------------------------------------------------------------
# Main layout — two columns
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("""
    <div style="padding:8px 0 20px 0;">
        <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">Upload Image</div>
        <div style="font-size:28px; font-weight:800; color:#1a1a1a; line-height:1.1; margin-bottom:6px;">Analyze a charger</div>
        <div style="font-size:14px; color:#666;">Upload a photo to detect component faults in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a charger image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Charger →")
    else:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e8e8e8; border-radius:12px; padding:40px 24px; text-align:center; margin-top:8px;">
            <div style="font-size:32px; margin-bottom:12px;">📷</div>
            <div style="font-size:15px; color:#1a1a1a; font-weight:600; margin-bottom:6px;">Upload a charger photo</div>
            <div style="font-size:13px; color:#999;">JPEG or PNG · Any resolution</div>
        </div>

        <div style="margin-top:24px;">
            <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:12px;">What FLARE detects</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        """, unsafe_allow_html=True)

        detect_items = [
            ("Screen", "Display damage, cracked glass"),
            ("Body / OOS", "Physical damage, out-of-service"),
            ("Cable", "Cable damage, fraying"),
            ("Plug", "Connector damage, missing plug"),
        ]
        cols = st.columns(2)
        for i, (title, desc) in enumerate(detect_items):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e8e8e8; border-radius:8px; padding:12px 14px; margin-bottom:8px;">
                    <div style="font-size:12px; font-weight:700; color:#1a1a1a; margin-bottom:3px;">{title}</div>
                    <div style="font-size:11px; color:#888;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        analyze_btn = False

with right_col:
    st.markdown("""
    <div style="padding:8px 0 20px 0;">
        <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">Results</div>
        <div style="font-size:28px; font-weight:800; color:#1a1a1a; line-height:1.1; margin-bottom:6px;">Fault Report</div>
        <div style="font-size:14px; color:#666;">Component-level classification with confidence scores.</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file and analyze_btn:
        with st.spinner("Running inference..."):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={"image": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                    data={"charger_id": charger_id or "DEMO"},
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    overall    = result.get("overall_status", "UNKNOWN")
                    latency    = result.get("processing_time_ms", 0)
                    flagged    = result.get("flagged_for_review", False)
                    components = result.get("components", {})

                    # Overall status card
                    if overall == "FAULT_DETECTED":
                        bg, border, title_color, icon, label = "#fef2f2", "#fca5a5", "#dc2626", "⚠", "Fault detected"
                    elif overall == "HEALTHY":
                        bg, border, title_color, icon, label = "#f0fdf4", "#86efac", "#16a34a", "✓", "All healthy"
                    else:
                        bg, border, title_color, icon, label = "#fffbeb", "#fde68a", "#d97706", "!", overall

                    st.markdown(f"""
                    <div style="background:{bg}; border:1.5px solid {border}; border-radius:12px; padding:20px 24px; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between;">
                        <div>
                            <div style="font-size:11px; color:#888; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Overall status</div>
                            <div style="font-size:24px; font-weight:800; color:{title_color};">{icon} {label.upper()}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:11px; color:#888; margin-bottom:4px;">Latency</div>
                            <div style="font-size:20px; font-weight:700; color:#1a1a1a;">{latency}ms</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if flagged:
                        st.markdown("""
                        <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:10px 14px; margin-bottom:16px;">
                            <span style="font-size:13px; color:#92400e;">⚠ One or more components flagged for human review (confidence &lt; 70%)</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Component cards
                    st.markdown("""
                    <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin:20px 0 12px 0;">Component analysis</div>
                    """, unsafe_allow_html=True)

                    card_cols = st.columns(2)
                    for i, (key, label) in enumerate(COMPONENT_LABELS.items()):
                        comp       = components.get(key, {})
                        detected   = comp.get("detected", False)
                        status     = comp.get("status")
                        confidence = comp.get("confidence")

                        with card_cols[i % 2]:
                            if not detected:
                                st.markdown(f"""
                                <div style="background:#ffffff; border:1px solid #e8e8e8; border-radius:10px; padding:18px; margin-bottom:10px;">
                                    <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">{label}</div>
                                    <div style="font-size:15px; color:#ccc; font-weight:600;">Not detected</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                is_broken    = status == "broken"
                                border_color = "#fca5a5" if is_broken else "#86efac"
                                left_border  = "#dc2626" if is_broken else "#16a34a"
                                status_color = "#dc2626" if is_broken else "#16a34a"
                                status_text  = "BROKEN" if is_broken else "HEALTHY"
                                conf_pct     = f"{confidence * 100:.1f}%" if confidence else "—"
                                low_conf     = confidence and confidence < 0.70
                                conf_color   = "#d97706" if low_conf else "#888"
                                flag_html    = '<div style="font-size:11px; color:#d97706; margin-top:4px;">⚠ flagged for review</div>' if low_conf else ""

                                st.markdown(f"""
                                <div style="background:#ffffff; border:1px solid {border_color}; border-left:4px solid {left_border}; border-radius:10px; padding:18px; margin-bottom:10px;">
                                    <div style="font-size:11px; color:#aaa; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">{label}</div>
                                    <div style="font-size:18px; font-weight:800; color:{status_color};">{status_text}</div>
                                    <div style="font-size:12px; color:{conf_color}; margin-top:6px;">Confidence: {conf_pct}</div>
                                    {flag_html}
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    with st.expander("View raw API response"):
                        st.json(result)

                else:
                    st.error(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.warning("Request timed out. Server may be cold starting — wait 30 seconds and try again.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    else:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e8e8e8; border-radius:12px; padding:60px 24px; text-align:center;">
            <div style="font-size:32px; margin-bottom:12px;">🔍</div>
            <div style="font-size:15px; color:#1a1a1a; font-weight:600; margin-bottom:6px;">Results will appear here</div>
            <div style="font-size:13px; color:#999;">Upload an image and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)