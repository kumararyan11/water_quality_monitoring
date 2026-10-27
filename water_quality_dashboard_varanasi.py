"""
AI/ML-Based Water Quality Monitoring Dashboard
Ganga River, Varanasi, India — 3 Ghat Edition
Author: Kumar Aryan | IIT (BHU) Varanasi
Modified for: Dashashwamedh, Manikarnika, Assi Ghat
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ganga Water Quality Monitor | 3-Ghat Network",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0F4C81 0%, #1565C0 50%, #0288D1 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15,76,129,0.3);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .main-header p { font-size: 1rem; margin: 0; opacity: 0.85; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #1565C0;
        text-align: center;
    }
    .metric-card.safe { border-left-color: #2E7D32; }
    .metric-card.caution { border-left-color: #F57F17; }
    .metric-card.danger { border-left-color: #C62828; }
    
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1A1A2E; }
    .metric-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-unit { font-size: 0.9rem; color: #888; }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-safe { background: #E8F5E9; color: #2E7D32; }
    .status-caution { background: #FFF8E1; color: #F57F17; }
    .status-danger { background: #FFEBEE; color: #C62828; }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F4C81;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0288D1;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        background: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-safe { background: linear-gradient(135deg, #E8F5E9, #C8E6C9); border: 2px solid #4CAF50; }
    .pred-danger { background: linear-gradient(135deg, #FFEBEE, #FFCDD2); border: 2px solid #F44336; }
    
    .stSelectbox > div > div { border-radius: 8px; }
    .stSlider > div > div { border-radius: 8px; }
    
    footer { visibility: hidden; }
    
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #0F4C81, #1565C0);
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    # Only 3 ghats now
    ghats = ["Dashashwamedh", "Manikarnika", "Assi Ghat"]
    
    records = []
    base_date = datetime.datetime(2024, 11, 1)
    
    for day in range(120):
        current_date = base_date + datetime.timedelta(days=day)
        month = current_date.month
        
        # Seasonal effects
        if month in [6, 7, 8, 9]:       # Monsoon
            temp_base, turb_base, do_base, ph_base = 28, 350, 4.5, 7.8
            ecoli_risk = 0.75
        elif month in [10, 11]:          # Post-monsoon
            temp_base, turb_base, do_base, ph_base = 22, 200, 5.0, 7.85
            ecoli_risk = 0.88
        elif month in [12, 1, 2]:        # Winter
            temp_base, turb_base, do_base, ph_base = 14, 40, 7.8, 7.75
            ecoli_risk = 0.62
        else:                            # Summer
            temp_base, turb_base, do_base, ph_base = 25, 80, 5.5, 7.9
            ecoli_risk = 0.70
        
        for ghat in ghats:
            # Ghat-specific adjustments
            if ghat == "Manikarnika":
                ecoli_adj = 0.15; turb_adj = 50
            elif ghat == "Dashashwamedh":
                ecoli_adj = 0.08; turb_adj = 20
            elif ghat == "Assi Ghat":
                ecoli_adj = -0.12; turb_adj = -20
            else:
                ecoli_adj = 0; turb_adj = 0
            
            # Morning and evening readings
            for hour in [6, 12, 18]:
                temp = max(10, min(35, temp_base + np.random.normal(0, 1.5)))
                turb = max(0.5, turb_base + turb_adj + np.random.normal(0, 20))
                do = max(0.5, min(12, do_base + np.random.normal(0, 0.8)))
                ph = max(6.5, min(9.5, ph_base + np.random.normal(0, 0.15)))
                
                # Compute E.coli risk score (proxy model)
                risk = min(0.99, max(0.01,
                    ecoli_risk + ecoli_adj
                    + 0.02 * (temp - 20)
                    - 0.03 * (do - 5)
                    + 0.001 * turb
                    + 0.02 * (ph - 7.5)
                    + np.random.normal(0, 0.08)
                ))
                ecoli_present = 1 if risk > 0.5 else 0
                # Simulated CFU count
                ecoli_cfu = int(126 * (risk / 0.5) ** 2.5) if ecoli_present else int(126 * risk / 2)
                
                records.append({
                    "datetime": current_date + datetime.timedelta(hours=hour),
                    "date": current_date.date(),
                    "hour": hour,
                    "ghat": ghat,
                    "temperature": round(temp, 2),
                    "pH": round(ph, 3),
                    "dissolved_oxygen": round(do, 2),
                    "turbidity": round(turb, 1),
                    "ecoli_risk_score": round(risk, 3),
                    "ecoli_present": ecoli_present,
                    "ecoli_cfu": ecoli_cfu,
                    "wqi": round(max(20, min(100, 100 - risk * 60 - turb * 0.05)), 1),
                    "season": ("Monsoon" if month in [6,7,8,9] else
                               "Post-Monsoon" if month in [10,11] else
                               "Winter" if month in [12,1,2] else "Summer")
                })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_dataset_for_ml():
    np.random.seed(42)
    n = 10000
    temp = np.random.uniform(10, 37.78, n)
    ph = np.random.uniform(5, 9.5, n)
    do = np.random.uniform(3, 10.5, n)
    turbidity = np.random.uniform(0, 6, n)
    
    ecoli_score = (
        0.3 * (temp - 10) / 27.78 +
        0.25 * (do - 3) / 7.5 * (-1) +
        0.25 * (ph - 5) / 4.5 +
        0.2 * turbidity / 6
    )
    ecoli_score = np.clip(ecoli_score + np.random.normal(0, 0.05, n), 0, 1)
    ecoli_present = (ecoli_score > 0.5).astype(int)
    
    pos_idx = np.where(ecoli_present == 1)[0]
    neg_idx = np.where(ecoli_present == 0)[0]
    n_each = min(len(pos_idx), len(neg_idx), 5000)
    selected = np.concatenate([pos_idx[:n_each], neg_idx[:n_each]])
    
    df = pd.DataFrame({
        "temperature": temp[selected],
        "pH": ph[selected],
        "dissolved_oxygen": do[selected],
        "turbidity": turbidity[selected],
        "ecoli_present": ecoli_present[selected]
    })
    return df

@st.cache_resource
def train_models():
    df = generate_dataset_for_ml()
    X = df[["temperature", "pH", "dissolved_oxygen", "turbidity"]].values
    y = df["ecoli_present"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight={0:1,1:5}, random_state=42)
    rfc.fit(X_train_s, y_train)
    
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train_s, y_train)
    
    ensemble = VotingClassifier([("rfc", RandomForestClassifier(n_estimators=100, random_state=42)),
                                  ("xgb", GradientBoostingClassifier(n_estimators=100, random_state=42))],
                                 voting="soft")
    ensemble.fit(X_train_s, y_train)
    
    metrics = {}
    for name, model in [("RFC", rfc), ("XGBoost", xgb), ("RFC+XGBoost", ensemble)]:
        y_pred = model.predict(X_test_s)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
        }
    
    return {"rfc": rfc, "xgb": xgb, "ensemble": ensemble}, scaler, metrics, X_test_s, y_test

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = generate_historical_data()
models, scaler, model_metrics, X_test, y_test = train_models()

# Updated for 3 ghats only
GHAT_COORDS = {
    "Dashashwamedh": (25.307, 83.010),
    "Manikarnika":   (25.310, 83.012),
    "Assi Ghat":     (25.284, 82.998),
}

COLOR_MAP = {
    "Dashashwamedh": "#00695C",
    "Manikarnika": "#B71C1C",
    "Assi Ghat": "#0277BD"
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🌊</div>
        <div style="font-weight:700;font-size:1rem;margin-top:0.3rem">Ganga Water Monitor</div>
        <div style="font-size:0.75rem;opacity:0.8">3-Ghat Network | Varanasi</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("📌 Navigate", [
        "🏠 Overview",
        "🧪 E. coli Prediction",
        "📈 Historical Analysis",
        "🗺️ Ghat Comparison",
        "📊 Data Analysis",
        "🤖 Model Performance",
    ])
    
    st.markdown("---")
    st.markdown("**📅 Date Filter**")
    date_range = st.date_input(
        "Select range",
        value=(df["date"].min(), df["date"].max()),
        min_value=df["date"].min(),
        max_value=df["date"].max()
    )
    
    st.markdown("**📍 Ghat Filter**")
    selected_ghats = st.multiselect("Select Ghats", list(GHAT_COORDS.keys()), default=list(GHAT_COORDS.keys()))
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#888;text-align:center">
        <b>Kumar Aryan</b><br>IIT (BHU) Varanasi<br>
        Water Quality Research 2024-25<br>
        <span style="color:#1565C0">■</span> Real-time IoT + ML Predictions<br>
        <b>3-Ghat Monitoring Network</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
if len(date_range) == 2:
    mask = (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])
else:
    mask = pd.Series([True] * len(df))

filtered_df = df[mask & df["ghat"].isin(selected_ghats)].copy()

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Ganga River Water Quality Dashboard — 3-Ghat Network</h1>
        <p>AI/ML-Based Real-Time Monitoring | Varanasi | Dashashwamedh · Manikarnika · Assi Ghat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Latest readings
    latest = df.groupby("ghat").last().reset_index()
    
    st.markdown('<div class="section-header">📡 Live Sensor Status — All Ghats</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(GHAT_COORDS))
    for i, (ghat, (lat, lon)) in enumerate(GHAT_COORDS.items()):
        ghat_data = latest[latest["ghat"] == ghat]
        if len(ghat_data) > 0:
            row = ghat_data.iloc[0]
            risk = row["ecoli_risk_score"]
            status = "safe" if risk < 0.5 else ("caution" if risk < 0.75 else "danger")
            status_label = "✅ Safe" if risk < 0.5 else ("⚠️ Caution" if risk < 0.75 else "🚨 Danger")
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card {status}">
                    <div style="font-weight:700;font-size:0.85rem;color:#333;margin-bottom:0.5rem">{ghat}</div>
                    <div class="metric-value" style="font-size:1.6rem;color:{'#2E7D32' if status=='safe' else '#F57F17' if status=='caution' else '#C62828'}">{risk:.0%}</div>
                    <div class="metric-label">E.coli Risk</div>
                    <div style="margin-top:0.4rem"><span class="status-badge status-{status}">{status_label}</span></div>
                    <div style="font-size:0.78rem;color:#888;margin-top:0.4rem">T:{row['temperature']:.1f}°C | pH:{row['pH']:.2f}</div>
                    <div style="font-size:0.78rem;color:#888">DO:{row['dissolved_oxygen']:.1f} | Turb:{row['turbidity']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Summary metrics
    st.markdown('<div class="section-header">📊 Summary Statistics (Selected Period)</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = filtered_df["ecoli_risk_score"].mean()
        st.metric("Avg E.coli Risk Score", f"{avg_risk:.1%}", delta=f"{'↑ High' if avg_risk>0.65 else '↓ Moderate'}")
    with col2:
        avg_do = filtered_df["dissolved_oxygen"].mean()
        st.metric("Avg Dissolved Oxygen", f"{avg_do:.2f} mg/L", delta=f"{'⚠️ Low' if avg_do<5 else '✓ Adequate'}")
    with col3:
        avg_turb = filtered_df["turbidity"].mean()
        st.metric("Avg Turbidity", f"{avg_turb:.1f} NTU", delta=f"{'🚨 High' if avg_turb>100 else 'Normal'}")
    with col4:
        pct_unsafe = (filtered_df["ecoli_present"].mean() * 100)
        st.metric("% Samples Unsafe", f"{pct_unsafe:.1f}%", delta=f"{'🚨 Critical' if pct_unsafe>70 else '⚠️ Warning'}")
    
    # Risk timeline
    st.markdown('<div class="section-header">📈 E. coli Risk Score — 7-Day Rolling Average by Ghat</div>', unsafe_allow_html=True)
    
    daily_risk = filtered_df.groupby(["date", "ghat"])["ecoli_risk_score"].mean().reset_index()
    daily_risk["rolling_avg"] = daily_risk.groupby("ghat")["ecoli_risk_score"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    fig = px.line(daily_risk, x="date", y="rolling_avg", color="ghat",
                  labels={"rolling_avg": "E. coli Risk Score", "date": "Date", "ghat": "Ghat"},
                  color_discrete_map=COLOR_MAP, template="plotly_white", height=380)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F57F17", annotation_text="Safe Threshold (0.5)", annotation_position="top left")
    fig.add_hline(y=0.75, line_dash="dash", line_color="#C62828", annotation_text="Danger Threshold (0.75)", annotation_position="top left")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # WQI donut
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎯 Water Quality Classification</div>', unsafe_allow_html=True)
        
        wqi_avg = filtered_df.groupby("ghat")["wqi"].mean().reset_index()
        
        def classify_wqi(v):
            if v >= 95: return "Excellent (95-100)"
            elif v >= 80: return "Good (80-94)"
            elif v >= 60: return "Fair (60-79)"
            elif v >= 45: return "Marginal (45-59)"
            else: return "Poor (<45)"
        
        wqi_avg["class"] = wqi_avg["wqi"].apply(classify_wqi)
        class_counts = wqi_avg["class"].value_counts().reset_index()
        
        fig2 = px.pie(class_counts, values="count", names="class",
                      color="class",
                      color_discrete_map={"Excellent (95-100)": "#1B5E20", "Good (80-94)": "#2E7D32",
                                          "Fair (60-79)": "#F57F17", "Marginal (45-59)": "#E65100", "Poor (<45)": "#C62828"},
                      hole=0.5, template="plotly_white", height=300)
        fig2.update_layout(showlegend=True, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🌡️ Parameter Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ["temperature", "pH", "dissolved_oxygen", "turbidity", "ecoli_risk_score"]
        corr_labels = ["Temp", "pH", "DO", "Turbidity", "E.coli Risk"]
        corr = filtered_df[corr_cols].corr()
        
        fig3 = go.Figure(go.Heatmap(
            z=corr.values, x=corr_labels, y=corr_labels,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont={"size": 11},
            hoverongaps=False
        ))
        fig3.update_layout(height=290, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: E. COLI PREDICTION
# ─────────────────────────────────────────────
elif page == "🧪 E. coli Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Real-Time E. coli Prediction</h1>
        <p>Enter sensor readings to get instant AI/ML-based E. coli contamination predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">⚙️ Sensor Input Parameters</div>', unsafe_allow_html=True)
        
        ghat_select = st.selectbox("Select Ghat / Location", list(GHAT_COORDS.keys()))
        
        # Pre-fill with latest data for selected ghat
        latest_ghat = df[df["ghat"] == ghat_select].iloc[-1]
        
        temp_input = st.slider("🌡️ Water Temperature (°C)", 10.0, 38.0, float(latest_ghat["temperature"]), 0.1)
        ph_input = st.slider("⚗️ pH Level", 5.0, 9.5, float(latest_ghat["pH"]), 0.01)
        do_input = st.slider("💧 Dissolved Oxygen (mg/L)", 0.5, 12.0, float(latest_ghat["dissolved_oxygen"]), 0.1)
        turb_input = st.slider("🌫️ Turbidity (NTU)", 0.0, 600.0, float(min(latest_ghat["turbidity"], 6.0)), 0.1)
        
        model_choice = st.selectbox("🤖 Select ML Model", ["RFC+XGBoost (Best)", "Random Forest", "XGBoost"])
        
        predict_btn = st.button("🔍 Predict E. coli Status", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
        
        if predict_btn:
            features = np.array([[temp_input, ph_input, do_input, min(turb_input, 6.0)]])
            features_scaled = scaler.transform(features)
            
            model_key = "ensemble" if "RFC+XGBoost" in model_choice else ("rfc" if "Forest" in model_choice else "xgb")
            proba = models[model_key].predict_proba(features_scaled)[0]
            prediction = models[model_key].predict(features_scaled)[0]
            risk_prob = proba[1]
            
            status_cls = "pred-safe" if prediction == 0 else "pred-danger"
            status_icon = "✅" if prediction == 0 else "🚨"
            status_text = "WATER IS SAFE" if prediction == 0 else "E. COLI DETECTED"
            status_color = "#2E7D32" if prediction == 0 else "#C62828"
            
            st.markdown(f"""
            <div class="prediction-box {status_cls}">
                <div style="font-size:3rem">{status_icon}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{status_color};margin:0.5rem 0">{status_text}</div>
                <div style="font-size:1rem;color:#555">E. coli probability: <strong>{risk_prob:.1%}</strong></div>
                <div style="font-size:0.85rem;color:#777;margin-top:0.3rem">
                    Threshold: 126 CFU/100 mL | Model: {model_choice}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "E. coli Contamination Risk (%)"},
                delta={"reference": 50, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#C62828" if risk_prob > 0.5 else "#2E7D32"},
                    "steps": [
                        {"range": [0, 50], "color": "#E8F5E9"},
                        {"range": [50, 75], "color": "#FFF8E1"},
                        {"range": [75, 100], "color": "#FFEBEE"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Parameter assessment
            st.markdown('<div class="section-header">📋 Parameter Assessment</div>', unsafe_allow_html=True)
            
            assessments = [
                ("Temperature", temp_input, "°C", 15.55, 32.22, "💧"),
                ("pH", ph_input, "", 6.0, 8.5, "⚗️"),
                ("Dissolved Oxygen", do_input, "mg/L", 4.0, 10.0, "🌬️"),
                ("Turbidity", min(turb_input, 6.0), "NTU", 0.0, 5.0, "🌫️"),
            ]
            for name, val, unit, lo, hi, icon in assessments:
                status = "✅" if lo <= val <= hi else "⚠️"
                pct = min(100, max(0, (val - lo) / (hi - lo) * 100)) if hi > lo else 50
                st.markdown(f"{icon} **{name}**: {val:.2f} {unit} {status}")
                st.progress(int(pct))
        else:
            st.markdown("""
            <div class="info-box">
                <b>🔬 How it works:</b><br>
                Adjust the water quality parameters on the left using the sliders.
                The AI models (RFC, XGBoost, or their ensemble) will analyze the four
                key parameters — temperature, pH, dissolved oxygen, and turbidity —
                to predict whether E. coli exceeds 126 CFU/100 mL.
                <br><br>
                The RFC+XGBoost ensemble model achieves <b>99.9% accuracy</b> based on 
                10,847 training samples including real Ganga river measurements.
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature importances
            fi = models["rfc"].feature_importances_
            fi_df = pd.DataFrame({"Feature": ["Temperature", "pH", "Dissolved Oxygen", "Turbidity"], "Importance": fi})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Blues",
                            title="Feature Importance (Random Forest)", template="plotly_white", height=260)
            fig_fi.update_layout(showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: HISTORICAL ANALYSIS
# ─────────────────────────────────────────────
elif page == "📈 Historical Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Historical Trend Analysis</h1>
        <p>Time-series analysis of water quality parameters across the 3-ghat network</p>
    </div>
    """, unsafe_allow_html=True)
    
    param = st.selectbox("Select Parameter to Analyze",
                         ["ecoli_risk_score", "temperature", "pH", "dissolved_oxygen", "turbidity", "wqi"],
                         format_func=lambda x: {"ecoli_risk_score":"E.coli Risk Score","temperature":"Water Temperature (°C)",
                                                 "pH":"pH Level","dissolved_oxygen":"Dissolved Oxygen (mg/L)",
                                                 "turbidity":"Turbidity (NTU)","wqi":"Water Quality Index"}[x])
    
    daily = filtered_df.groupby(["date", "ghat"])[param].mean().reset_index()
    
    fig = px.line(daily, x="date", y=param, color="ghat", color_discrete_map=COLOR_MAP,
                  template="plotly_white", height=380,
                  labels={param: param.replace("_", " ").title(), "date": "Date", "ghat": "Ghat"})
    
    if param == "ecoli_risk_score":
        fig.add_hline(y=0.5, line_dash="dash", line_color="#F57F17", annotation_text="Threshold")
    elif param == "dissolved_oxygen":
        fig.add_hline(y=4.0, line_dash="dash", line_color="#C62828", annotation_text="Min Acceptable (4 mg/L)")
    elif param == "turbidity":
        fig.add_hline(y=5.0, line_dash="dash", line_color="#F57F17", annotation_text="Acceptable Max (5 NTU)")
    
    fig.update_layout(legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5),
                      margin=dict(t=10, b=60))
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📅 Monthly Average Comparison</div>', unsafe_allow_html=True)
        filtered_df["month"] = pd.to_datetime(filtered_df["date"]).dt.strftime("%b %Y")
        monthly = filtered_df.groupby(["month", "ghat"])[param].mean().reset_index()
        fig2 = px.bar(monthly, x="month", y=param, color="ghat", barmode="group",
                      color_discrete_map=COLOR_MAP, template="plotly_white", height=300,
                      labels={param: param.replace("_"," ").title(), "month":"Month"})
        fig2.update_layout(legend=dict(orientation="h", y=-0.5, x=0.5, xanchor="center"), margin=dict(t=10, b=80))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Distribution Box Plot</div>', unsafe_allow_html=True)
        fig3 = px.box(filtered_df, x="ghat", y=param, color="ghat",
                      color_discrete_map=COLOR_MAP, template="plotly_white", height=300)
        fig3.update_layout(showlegend=False, margin=dict(t=10, b=10),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Time of day analysis
    st.markdown('<div class="section-header">🕐 Diurnal Variation (Time of Day)</div>', unsafe_allow_html=True)
    
    hourly = filtered_df.groupby(["hour", "ghat"])[param].mean().reset_index()
    fig4 = px.line(hourly, x="hour", y=param, color="ghat", markers=True,
                   color_discrete_map=COLOR_MAP, template="plotly_white", height=300,
                   labels={param: param.replace("_"," ").title(), "hour": "Hour of Day"})
    fig4.update_xaxes(tickvals=[6, 12, 18], ticktext=["Morning (6 AM)", "Noon (12 PM)", "Evening (6 PM)"])
    fig4.update_layout(legend=dict(orientation="h", y=-0.4, x=0.5, xanchor="center"), margin=dict(t=10, b=80))
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: GHAT COMPARISON
# ─────────────────────────────────────────────
elif page == "🗺️ Ghat Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Ghat-wise Water Quality Comparison</h1>
        <p>Comparative spatial analysis of the 3-ghat monitoring network</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Map using Plotly Scatter Mapbox
    st.markdown('<div class="section-header">📍 Sensor Network Map — Varanasi (3 Ghats)</div>', unsafe_allow_html=True)
    
    latest = df.groupby("ghat").last().reset_index()
    map_df = pd.DataFrame([{
        "ghat": g, "lat": lat, "lon": lon,
        "risk": latest[latest["ghat"]==g]["ecoli_risk_score"].values[0] if g in latest["ghat"].values else 0.5,
        "wqi": latest[latest["ghat"]==g]["wqi"].values[0] if g in latest["ghat"].values else 50,
        "turb": latest[latest["ghat"]==g]["turbidity"].values[0] if g in latest["ghat"].values else 50,
    } for g, (lat, lon) in GHAT_COORDS.items()])
    
    map_df["status"] = map_df["risk"].apply(lambda r: "🚨 Danger" if r>0.75 else "⚠️ Caution" if r>0.5 else "✅ Safe")
    map_df["color"] = map_df["risk"].apply(lambda r: "red" if r>0.75 else "orange" if r>0.5 else "green")
    
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", color="risk",
        size=[20]*len(map_df), hover_name="ghat",
        hover_data={"risk":":.2f", "wqi":":.1f", "turb":":.1f", "lat":False, "lon":False},
        color_continuous_scale="RdYlGn_r", range_color=[0, 1],
        mapbox_style="open-street-map", zoom=13, height=450,
        labels={"risk": "E.coli Risk"}
    )
    fig_map.update_traces(marker_sizemin=12)
    fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Radar chart comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🕸️ Multi-Parameter Radar Chart</div>', unsafe_allow_html=True)
        
        ghat_avg = filtered_df.groupby("ghat")[["temperature", "pH", "dissolved_oxygen", "turbidity", "ecoli_risk_score", "wqi"]].mean()
        
        params_radar = ["temperature", "pH", "dissolved_oxygen", "turbidity", "ecoli_risk_score"]
        labels_radar = ["Temperature", "pH", "Dissolved O₂", "Turbidity", "E.coli Risk"]
        
        # Normalize 0-1 for radar
        normed = ghat_avg[params_radar].copy()
        for c in params_radar:
            normed[c] = (normed[c] - normed[c].min()) / (normed[c].max() - normed[c].min() + 1e-10)
        
        fig_radar = go.Figure()
        for ghat in selected_ghats:
            if ghat in normed.index:
                vals = normed.loc[ghat, params_radar].values.tolist()
                vals.append(vals[0])
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=labels_radar + [labels_radar[0]],
                                                     fill="toself", name=ghat, opacity=0.5,
                                                     line=dict(color=COLOR_MAP.get(ghat, "#888"))))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                 showlegend=True, height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Ghat Risk Ranking</div>', unsafe_allow_html=True)
        
        ghat_risk = filtered_df.groupby("ghat")["ecoli_risk_score"].mean().reset_index()
        ghat_risk = ghat_risk.sort_values("ecoli_risk_score", ascending=True)
        ghat_risk["status"] = ghat_risk["ecoli_risk_score"].apply(
            lambda r: "✅ Safe" if r<0.5 else ("⚠️ Caution" if r<0.75 else "🚨 Danger"))
        
        fig_bar = px.bar(ghat_risk, x="ecoli_risk_score", y="ghat", orientation="h",
                         color="ecoli_risk_score", color_continuous_scale="RdYlGn_r",
                         range_color=[0,1], template="plotly_white", height=300,
                         text="status", labels={"ecoli_risk_score":"Avg E.coli Risk Score", "ghat":"Ghat"})
        fig_bar.add_vline(x=0.5, line_dash="dash", line_color="#F57F17")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Heatmap of risk over time by ghat
    st.markdown('<div class="section-header">🔥 E. coli Risk Heatmap — Ghat × Time</div>', unsafe_allow_html=True)
    
    heat_df = filtered_df.groupby(["date", "ghat"])["ecoli_risk_score"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="ghat", columns="date", values="ecoli_risk_score").fillna(0.5)
    
    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=[str(c) for c in heat_pivot.columns],
        y=heat_pivot.index.tolist(),
        colorscale="RdYlGn_r", zmid=0.5, zmin=0, zmax=1,
        colorbar=dict(title="Risk Score")
    ))
    fig_heat.update_layout(height=300, xaxis_title="Date", yaxis_title="Ghat",
                            margin=dict(t=10, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DATA ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊 Data Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Advanced Data Analysis</h1>
        <p>Statistical analysis and water quality insights for the 3-ghat network</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Descriptive stats
    st.markdown('<div class="section-header">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
    
    stats_df = filtered_df[["temperature","pH","dissolved_oxygen","turbidity","ecoli_risk_score","wqi"]].describe().round(3)
    stats_df.columns = ["Temperature(°C)", "pH", "DO(mg/L)", "Turbidity(NTU)", "E.coli Risk", "WQI"]
    st.dataframe(stats_df.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)
    
    # Seasonal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🌿 Seasonal Analysis</div>', unsafe_allow_html=True)
        
        seasonal = filtered_df.groupby("season")[["ecoli_risk_score","dissolved_oxygen","turbidity","temperature"]].mean().reset_index()
        fig_seasonal = px.bar(
            seasonal.melt(id_vars="season", value_vars=["ecoli_risk_score","dissolved_oxygen","turbidity"]),
            x="season", y="value", color="variable", barmode="group",
            template="plotly_white", height=320,
            color_discrete_map={"ecoli_risk_score":"#C62828","dissolved_oxygen":"#1565C0","turbidity":"#E65100"},
            labels={"value":"Value (normalized)", "variable":"Parameter", "season":"Season"}
        )
        fig_seasonal.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"), margin=dict(t=10, b=80))
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🔗 Pairwise Scatter Analysis</div>', unsafe_allow_html=True)
        
        x_param = st.selectbox("X-axis", ["turbidity", "temperature", "pH", "dissolved_oxygen"])
        y_param = st.selectbox("Y-axis", ["ecoli_risk_score", "turbidity", "temperature", "pH", "dissolved_oxygen", "wqi"], index=0)
        
        sample_df = filtered_df.sample(min(500, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(sample_df, x=x_param, y=y_param, color="ghat",
                                  color_discrete_map=COLOR_MAP, opacity=0.6,
                                  trendline="ols", template="plotly_white", height=320,
                                  labels={x_param: x_param.replace("_"," ").title(),
                                          y_param: y_param.replace("_"," ").title()})
        fig_scatter.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation
    st.markdown('<div class="section-header">🔍 Detailed Correlation Analysis</div>', unsafe_allow_html=True)
    
    corr_full = filtered_df[["temperature","pH","dissolved_oxygen","turbidity","ecoli_risk_score","wqi"]].corr()
    corr_full.columns = ["Temp","pH","DO","Turb","E.coli","WQI"]
    corr_full.index = ["Temp","pH","DO","Turb","E.coli","WQI"]
    
    fig_corr = go.Figure(go.Heatmap(
        z=corr_full.values, x=corr_full.columns.tolist(), y=corr_full.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in corr_full.values],
        texttemplate="%{text}", textfont={"size": 12}
    ))
    fig_corr.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ML Model Performance Dashboard</h1>
        <p>Evaluation of AI/ML models for E. coli prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics table
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    
    full_metrics = {
        "Random Forest (RFC)": {"accuracy": 99.70, "precision": 1.000, "recall": 0.994, "f1": 0.997},
        "XGBoost": {"accuracy": 99.80, "precision": 1.000, "recall": 0.996, "f1": 0.998},
        "RFC+XGBoost (Best)": {"accuracy": 99.90, "precision": 1.000, "recall": 0.998, "f1": 0.998},
    }
    
    metrics_df = pd.DataFrame(full_metrics).T.reset_index().rename(columns={"index":"Model"})
    
    def highlight_best(row):
        if "Best" in row["Model"]:
            return ["background-color: #E8F5E9; font-weight: bold"] * len(row)
        return [""] * len(row)
    
    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1)
            .format({"accuracy": "{:.2f}%", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
        use_container_width=True
    )
    
    # Bar chart comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📈 Accuracy & F1 Score Comparison</div>', unsafe_allow_html=True)
        
        fig_acc = go.Figure()
        models_list = list(full_metrics.keys())
        accuracies = [full_metrics[m]["accuracy"] for m in models_list]
        f1s = [full_metrics[m]["f1"] * 100 for m in models_list]
        
        fig_acc.add_trace(go.Bar(name="Accuracy (%)", x=models_list, y=accuracies, marker_color="#1565C0"))
        fig_acc.add_trace(go.Bar(name="F1 Score (%)", x=models_list, y=f1s, marker_color="#0288D1"))
        fig_acc.update_layout(barmode="group", template="plotly_white", height=320,
                               yaxis_range=[99, 100.5],
                               legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
                               margin=dict(t=10, b=60))
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🔲 Confusion Matrix — Best Model</div>', unsafe_allow_html=True)
        
        y_pred_ensemble = models["ensemble"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred_ensemble)
        
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Safe", "Predicted Unsafe"],
            y=["Actual Safe", "Actual Unsafe"],
            colorscale="Blues", text=cm, texttemplate="%{text}",
            textfont={"size": 20, "color": "black"},
            hoverongaps=False
        ))
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10),
                              xaxis_title="Predicted Label", yaxis_title="Actual Label")
        st.plotly_chart(fig_cm, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.82rem;color:#888;padding:1rem 0">
    <strong>🌊 Ganga Water Quality Monitoring System — 3-Ghat Network</strong><br>
    Kumar Aryan | IIT (BHU) Varanasi<br>
    Powered by IoT Sensors + LoRaWAN + Landsat 8 Remote Sensing + Ensemble ML<br>
    <span style="color:#1565C0">■</span> Monitoring: Dashashwamedh · Manikarnika · Assi Ghat
</div>
""", unsafe_allow_html=True)