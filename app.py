import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="wide"
)

# -------------------- GLOBAL CSS FIX --------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background:
        linear-gradient(rgba(5,15,25,0.88), rgba(5,15,25,0.88)),
        url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-attachment: fixed;
}

/* All text visibility */
h1, h2, h3, h4, h5, h6, p, li, span, label {
    color: #ffffff !important;
}

/* Dropdown visibility */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Slider labels */
.stSlider label {
    color: #ffffff !important;
}

/* Cards */
.card {
    background: rgba(20,30,45,0.9);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}

/* Section spacing */
.section {
    margin-top: 35px;
}

/* Plot containers */
.plot-card {
    background: rgba(255,255,255,0.95);
    border-radius: 14px;
    padding: 10px;
}
/* Fix Predict button text visibility */
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #022c22 !important;   /* FORCE TEXT COLOR */
    font-weight: 800 !important;
    font-size: 16px;
    padding: 12px 34px;
    border-radius: 14px;
    border: none;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ===== FIX SELECTBOX VISIBILITY ===== */

/* Closed selectbox */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px;
    font-weight: 600;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #ffffff !important;
}

/* Dropdown options */
li[role="option"] {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Hover effect */
li[role="option"]:hover {
    background-color: #e5e7eb !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------
model = pickle.load(open("catboost_model.pkl", "rb"))

FEATURE_ORDER = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalch','exang','oldpeak','slope','ca','thal'
]

# -------------------- TITLE --------------------
st.markdown("## ❤️ Heart Disease Risk Assessment")
st.markdown("AI-assisted cardiovascular risk evaluation based on clinical indicators")

# -------------------- INPUT LAYOUT --------------------
left, right = st.columns(2)

with left:
    st.markdown("### 🔽 Clinical Selections")
    sex = st.selectbox("Sex", ["Male (1)", "Female (0)"])
    cp = st.selectbox("Chest Pain Type", [
        "0 – Typical Angina",
        "1 – Atypical Angina",
        "2 – Non-anginal Pain",
        "3 – Asymptomatic"
    ])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0 – No", "1 – Yes"])
    restecg = st.selectbox("Resting ECG", [
        "0 – Normal",
        "1 – ST-T abnormality",
        "2 – LV Hypertrophy"
    ])
    exang = st.selectbox("Exercise Induced Angina", ["0 – No", "1 – Yes"])
    slope = st.selectbox("Slope of ST Segment", [
        "0 – Upsloping",
        "1 – Flat",
        "2 – Downsloping"
    ])
    ca = st.selectbox("Major Vessels", ["0", "1", "2", "3"])
    thal = st.selectbox("Thalassemia", [
        "1 – Normal",
        "2 – Fixed Defect",
        "3 – Reversible Defect"
    ])

with right:
    st.markdown("### 📊 Measured Values")
    age = st.slider("Age (years)", 20, 80, 50)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
    thalch = st.slider("Maximum Heart Rate", 60, 220, 150)
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([[
        age,
        int(sex[-2]),
        int(cp[0]),
        trestbps,
        chol,
        int(fbs[0]),
        int(restecg[0]),
        thalch,
        int(exang[0]),
        oldpeak,
        int(slope[0]),
        int(ca),
        int(thal[0])
    ]], columns=FEATURE_ORDER)

    prob = model.predict_proba(input_df)[0][1] * 100

    if prob < 35:
        level = "LOW RISK"
        color = "#2ecc71"
    elif prob < 65:
        level = "MODERATE RISK"
        color = "#f1c40f"
    else:
        level = "SEVERE RISK"
        color = "#e74c3c"
    st.markdown("## 🧾 Entered Patient Details")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown(f"""
        **Age:** {age} years  
        **Sex:** {sex}  
        **Chest Pain Type:** {cp}  
        **Resting Blood Pressure:** {trestbps} mmHg  
        **Cholesterol:** {chol} mg/dl  
        **Fasting Blood Sugar > 120:** {fbs}  
        """)

    with d2:
        st.markdown(f"""
        **Max Heart Rate (Thalach):** {thalch}  
        **Exercise Induced Angina:** {exang}  
        **ST Depression (Oldpeak):** {oldpeak}  
        **ST Slope:** {slope}  
        **Major Vessels (CA):** {ca}  
        **Thalassemia:** {thal}  
        """)

    # -------------------- RESULT --------------------
    st.markdown(f"""
    <div class="card">
    <h2 style="color:{color}">Predicted Risk Level: {level}</h2>
    <h3>Risk Probability: {prob:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- INTERPRETATION (FIXED VISIBILITY) --------------------
    st.markdown("### 🫀 Interpretation of Your Condition")

    st.markdown("""
    **Normal Heart Condition Typically Shows:**
    - Resting blood pressure around **120 mmHg**
    - Cholesterol levels below **200 mg/dl**
    - Strong heart rate response during exercise
    - Absence of exercise-induced angina
    - No major vessel blockage

    **Your Predicted Condition Indicates:**
    - Patterns that deviate from optimal cardiovascular ranges
    - Stress response indicators affecting heart workload
    - Possible vessel-related or ECG-linked risk markers

    **What This Means:**
    This AI-based assessment highlights statistical risk patterns.
    It does **not** confirm a medical diagnosis, but it strongly
    suggests whether preventive or clinical attention may be needed.
    """)

    # -------------------- VISUALIZATIONS (REPLACED & IMPROVED) --------------------
    st.markdown("## 📊 Visual Risk Analysis")

    # =======================
    # ROW 1
    # =======================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: RISK GAUGE ----------
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title=dict(
                text="Overall Risk (%)",
                font=dict(color="#1498b6", size=18)
            ),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#2563eb"),
                steps=[
                    dict(range=[0, 35], color="#d1fae5"),
                    dict(range=[35, 65], color="#fef3c7"),
                    dict(range=[65, 100], color="#fee2e2")
                ]
            )
        ))
        fig1.update_layout(
            height=260,
            paper_bgcolor="#031C36",
            plot_bgcolor="#647da9"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ---------- PLOT 2: RISK DISTRIBUTION ----------
    with col2:
        fig2 = go.Figure(go.Pie(
            labels=["Healthy", "At Risk"],
            values=[100 - prob, prob],
            hole=0.55,
            marker=dict(colors=["#22c55e", "#dc2626"]),
            textinfo="label+percent"
        ))
        fig2.update_layout(
            title=dict(
                text="Risk Distribution",
                font=dict(color="#1498b6", size=18)
            ),
            height=260,
            paper_bgcolor="#031C36",
            plot_bgcolor="#647da9",
            legend=dict(font=dict(color="black"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    # =======================
    # ROW 2
    # =======================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: CHOLESTEROL COMPARISON ----------
    with col3:
        fig3 = go.Figure(go.Bar(
            x=["Normal Cholesterol", "Your Cholesterol"],
            y=[200, chol],
            marker=dict(color=["#94a3b8", "#2563eb"]),
            text=[200, chol],
            textposition="outside"
        ))
        fig3.update_layout(
            title=dict(
                text="Cholesterol Comparison (mg/dl)",
                font=dict(color="#1498b6", size=18)
            ),
            height=260,
            paper_bgcolor="#031C36",
            plot_bgcolor="#647da9",
            yaxis=dict(
                title="mg/dl",
                gridcolor="#e5e7eb"
            ),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ---------- PLOT 4: BLOOD PRESSURE STRESS ----------
    with col4:
        fig4 = go.Figure(go.Scatter(
            x=["Rest", "Stress"],
            y=[trestbps, trestbps + oldpeak * 10],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=10, color="#2563eb"),
            name="Blood Pressure"
        ))
        fig4.update_layout(
            title=dict(
                text="Blood Pressure Stress Response",
                font=dict(color="#1498b6", size=18)
            ),
            height=260,
            paper_bgcolor="#031C36",
            plot_bgcolor="#647da9",
            yaxis=dict(
                title="mmHg",
                gridcolor="#e5e7eb"
            ),
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

    # -------------------- FOOTER --------------------
    st.markdown("""
    ---
    **Disclaimer:** This tool is for educational purposes only.  
    Always consult a qualified healthcare professional for diagnosis.
    """)
