import streamlit as st
import pandas as pd
import joblib
import time
import requests
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Stroke AI",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- LOTTIE FUNCTION ---------------- #
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

heart_lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #12001a, #2b0036, #40004d, #1a001a);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px 0 rgba(255, 0, 150, 0.3);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff004f, #8000ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 25px;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 25px rgba(255,0,100,0.7);
}

/* Heart Beat Animation */
.heartbeat {
    font-size: 45px;
    animation: beat 1s infinite;
    display: inline-block;
}

@keyframes beat {
    0% {transform: scale(1);}
    25% {transform: scale(1.2);}
    40% {transform: scale(1);}
    60% {transform: scale(1.2);}
    100% {transform: scale(1);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<h1>AI Heart Stroke Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='heartbeat'>❤️</div>", unsafe_allow_html=True)

with col2:
    st_lottie(heart_lottie, height=180)

# ---------------- GLASS CONTAINER ---------------- #
st.markdown("<div class='glass'>", unsafe_allow_html=True)

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- PREDICT ---------------- #
if st.button("🔮 Analyze Risk"):

    with st.spinner("Analyzing heart patterns..."):
        time.sleep(2)

        raw_data = {
            "Age": age,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingBP": resting_bp,
            "Cholesterol": cholestrol,
            "FastingBS": fasting_bs,
            "RestingECG_" + resting_ecg: 1,
            "MaxHR": max_hr,
            "ExerciseAngina_" + exercise_angina: 1,
            "Oldpeak": oldpeak,
            "ST_Slope_" + st_slope: 1
        }

        input_data = pd.DataFrame([raw_data])

        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_columns]
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100

    st.markdown("---")

    # --------- AUTO COLOR SCALING --------- #
    if probability < 35:
        color = "green"
        status = "LOW RISK"
    elif probability < 70:
        color = "orange"
        status = "MODERATE RISK"
    else:
        color = "red"
        status = "HIGH RISK"

    st.markdown(
        f"""
        <div style="
            background: rgba(0,0,0,0.6);
            padding:20px;
            border-radius:15px;
            border:2px solid {color};
            text-align:center;
            font-size:22px;">
            <h2 style='color:{color};'>{status}</h2>
            <p>Risk Probability: <b>{probability:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
