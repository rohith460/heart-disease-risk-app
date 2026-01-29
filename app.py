import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    FIG_SIZE = (3.6, 2.6)
    FACE_COLOR = "#1a4a7a"
    TITLE_COLOR = "#b0c5ff"

    # =======================
    # ROW 1
    # =======================
    col1, col2 = st.columns(2)

    # ---------- PLOT 1: SPEEDOMETER / RISK METER ----------
    with col1:
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        fig1.patch.set_facecolor(FACE_COLOR)
        ax1.set_facecolor(FACE_COLOR)

        theta = np.linspace(np.pi, 0, 200)
        ax1.plot(np.cos(theta), np.sin(theta), linewidth=8, color="#94a3b8")

        ax1.plot(np.cos(theta[:70]), np.sin(theta[:70]), linewidth=8, color="#22c55e")
        ax1.plot(np.cos(theta[70:130]), np.sin(theta[70:130]), linewidth=8, color="#facc15")
        ax1.plot(np.cos(theta[130:]), np.sin(theta[130:]), linewidth=8, color="#dc2626")

        angle = np.pi * (1 - prob / 100)
        ax1.plot(
            [0, 0.85 * np.cos(angle)],
            [0, 0.85 * np.sin(angle)],
            linewidth=3,
            color="#60a5fa"
        )

        ax1.scatter(0, 0, s=60, color="#60a5fa")
        ax1.text(0, -0.2, f"{prob:.1f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=TITLE_COLOR)

        ax1.set_title("Overall Risk Meter", color=TITLE_COLOR, fontsize=12)
        ax1.axis("off")
        ax1.set_aspect("equal")

        plt.tight_layout()
        st.pyplot(fig1)

    # ---------- PLOT 2: RISK DISTRIBUTION ----------
    with col2:
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        fig2.patch.set_facecolor(FACE_COLOR)
        ax2.set_facecolor(FACE_COLOR)

        ax2.pie(
            [100 - prob, prob],
            labels=["Healthy", "At Risk"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#22c55e", "#dc2626"],
            wedgeprops=dict(width=0.45),
            textprops=dict(color="white", fontsize=9)
        )

        ax2.set_title("Risk Distribution", color=TITLE_COLOR, fontsize=12)
        ax2.axis("equal")

        plt.tight_layout()
        st.pyplot(fig2)

    # =======================
    # ROW 2
    # =======================
    col3, col4 = st.columns(2)

    # ---------- PLOT 3: CHOLESTEROL COMPARISON ----------
    with col3:
        fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
        fig3.patch.set_facecolor(FACE_COLOR)
        ax3.set_facecolor(FACE_COLOR)

        ax3.bar(
            ["Normal", "Your Value"],
            [200, chol],
            color=["#94a3b8", "#60a5fa"]
        )

        ax3.axhline(200, linestyle="--", color="#e5e7eb", linewidth=1)
        ax3.set_ylabel("mg/dl", color="white")
        ax3.set_title("Cholesterol Comparison", color=TITLE_COLOR, fontsize=12)

        ax3.tick_params(colors="white")
        ax3.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig3)

    # ---------- PLOT 4: BLOOD PRESSURE STRESS ----------
    with col4:
        fig4, ax4 = plt.subplots(figsize=FIG_SIZE)
        fig4.patch.set_facecolor(FACE_COLOR)
        ax4.set_facecolor(FACE_COLOR)

        ax4.plot(
            ["Rest", "Stress"],
            [trestbps, trestbps + oldpeak * 10],
            marker="o",
            linewidth=2.2,
            color="#60a5fa"
        )

        ax4.set_ylabel("mmHg", color="white")
        ax4.set_title("Blood Pressure Stress Response", color=TITLE_COLOR, fontsize=12)

        ax4.tick_params(colors="white")
        ax4.grid(alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig4)



    # -------------------- FOOTER --------------------
    st.markdown("""
    ---
    **Disclaimer:** This tool is for educational purposes only.  
    Always consult a qualified healthcare professional for diagnosis.
    """)
